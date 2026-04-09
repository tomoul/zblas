// zblas/src/level3/sgemm_bias.zig
//
// Fused SGEMM + row-broadcast bias addition:
//   C[i,j] = sum_k A[i,k]*B[k,j] + bias[j]
//
// Eliminates the separate bias memory pass by adding bias in the SGEMM store
// epilogue while the result is still in registers. The sentence transformer
// calls matmul+bias 36× per forward pass (6 layers × 6 ops/layer), so this
// fusion avoids 36 extra sweeps over the output matrix.
//
// The kernel specializes for the skinny-M case (M ≤ SKINNY_M_THRESHOLD) which
// covers all sentence transformer inference shapes (M = seq_len ≤ 64 tokens).
// For larger M, falls back to standard sgemm + scalar bias loop.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../config.zig");
const sgemm_impl = @import("sgemm.zig");

/// Fused SGEMM + bias: C = A*B + broadcast(bias)
///
/// A: [M × K], B: [K × N], bias: [N], C: [M × N]
/// Each row of C gets the same bias vector added.
pub fn sgemmBias(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    bias: [*]const f32,
    C: []f32,
) void {
    if (M == 0 or N == 0 or K == 0) return;

    // Skinny-M path: fused bias in store epilogue
    if (M <= config.SKINNY_M_THRESHOLD) {
        const use_skinny = (M <= 4) or (M % 4 != 0);
        if (use_skinny) {
            sgemmSkinnyBiasDispatch(M, N, K, A, K, B, N, bias, C, N);
            return;
        }
    }

    // Fallback: standard sgemm then add bias in-place
    sgemm_impl.sgemm(M, N, K, A, B, C, 1.0, 0.0);
    addBiasRows(M, N, bias, C, N);
}

/// SGEMM + bias with leading dimensions (for non-packed submatrix ops)
pub fn sgemmBiasLd(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    bias: [*]const f32,
    C: []f32,
    ldc: usize,
) void {
    if (M == 0 or N == 0 or K == 0) return;

    if (M <= config.SKINNY_M_THRESHOLD) {
        const use_skinny = (M <= 4) or (M % 4 != 0);
        if (use_skinny) {
            sgemmSkinnyBiasDispatch(M, N, K, A, lda, B, ldb, bias, C, ldc);
            return;
        }
    }

    sgemm_impl.sgemmTranspose(
        @import("../zblas.zig").Transpose.NoTrans,
        @import("../zblas.zig").Transpose.NoTrans,
        M, N, K, A, lda, B, ldb, C, ldc, 1.0, 0.0,
    );
    addBiasRows(M, N, bias, C, ldc);
}

// ============================================================================
// Skinny-M fused kernel (M ≤ SKINNY_M_THRESHOLD)
// ============================================================================

fn sgemmSkinnyBiasDispatch(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    bias: [*]const f32,
    C: []f32,
    ldc: usize,
) void {
    comptime var rows: usize = 1;
    inline while (rows <= config.SKINNY_M_THRESHOLD) : (rows += 1) {
        if (M == rows) {
            sgemmSkinnyBiasKernel(rows, N, K, A, lda, B, ldb, bias, C, ldc);
            return;
        }
    }
}

fn sgemmSkinnyBiasKernel(
    comptime ROWS: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    bias: [*]const f32,
    C: []f32,
    ldc: usize,
) void {
    const VEC_WIDTH = comptime config.getVectorWidth();
    const Vec = @Vector(VEC_WIDTH, f32);

    var j: usize = 0;

    // Tier 1: M ≤ 4 → 3×VEC_WIDTH columns (24 cols on AVX2)
    if (comptime ROWS <= 4) {
        const NR = 3 * VEC_WIDTH;
        while (j + NR <= N) : (j += NR) {
            var acc0: [ROWS]Vec = undefined;
            var acc1: [ROWS]Vec = undefined;
            var acc2: [ROWS]Vec = undefined;
            inline for (0..ROWS) |r| {
                acc0[r] = @splat(0.0);
                acc1[r] = @splat(0.0);
                acc2[r] = @splat(0.0);
            }
            for (0..K) |k| {
                const b0: Vec = B[k * ldb + j ..][0..VEC_WIDTH].*;
                const b1: Vec = B[k * ldb + j + VEC_WIDTH ..][0..VEC_WIDTH].*;
                const b2: Vec = B[k * ldb + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].*;
                inline for (0..ROWS) |r| {
                    const a: Vec = @splat(A[r * lda + k]);
                    acc0[r] += a * b0;
                    acc1[r] += a * b1;
                    acc2[r] += a * b2;
                }
            }
            // Fused bias: load once, add to all rows
            const bias0: Vec = bias[j..][0..VEC_WIDTH].*;
            const bias1: Vec = bias[j + VEC_WIDTH ..][0..VEC_WIDTH].*;
            const bias2: Vec = bias[j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].*;
            inline for (0..ROWS) |r| {
                C[r * ldc + j ..][0..VEC_WIDTH].* = acc0[r] + bias0;
                C[r * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH].* = acc1[r] + bias1;
                C[r * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = acc2[r] + bias2;
            }
        }
    }

    // Tier 2: single VEC_WIDTH (8 cols on AVX2)
    while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
        var acc: [ROWS]Vec = undefined;
        inline for (0..ROWS) |r| {
            acc[r] = @splat(0.0);
        }
        for (0..K) |k| {
            const b_vec: Vec = B[k * ldb + j ..][0..VEC_WIDTH].*;
            inline for (0..ROWS) |r| {
                acc[r] += @as(Vec, @splat(A[r * lda + k])) * b_vec;
            }
        }
        const bias_vec: Vec = bias[j..][0..VEC_WIDTH].*;
        inline for (0..ROWS) |r| {
            C[r * ldc + j ..][0..VEC_WIDTH].* = acc[r] + bias_vec;
        }
    }

    // Scalar tail
    while (j < N) : (j += 1) {
        var acc: [ROWS]f32 = undefined;
        inline for (0..ROWS) |r| {
            acc[r] = 0.0;
        }
        for (0..K) |k| {
            const b_val = B[k * ldb + j];
            inline for (0..ROWS) |r| {
                acc[r] += A[r * lda + k] * b_val;
            }
        }
        const bias_val = bias[j];
        inline for (0..ROWS) |r| {
            C[r * ldc + j] = acc[r] + bias_val;
        }
    }
}

// ============================================================================
// Bias add fallback (for non-skinny path)
// ============================================================================

fn addBiasRows(
    M: usize,
    N: usize,
    bias: [*]const f32,
    C: []f32,
    ldc: usize,
) void {
    const VEC_WIDTH = comptime config.getVectorWidth();
    const Vec = @Vector(VEC_WIDTH, f32);

    for (0..M) |row| {
        const row_start = row * ldc;
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            const c_ptr = C[row_start + j ..][0..VEC_WIDTH];
            const bias_vec: Vec = bias[j..][0..VEC_WIDTH].*;
            c_ptr.* = @as(Vec, c_ptr.*) + bias_vec;
        }
        while (j < N) : (j += 1) {
            C[row_start + j] += bias[j];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "sgemmBias basic 2x3" {
    // A = [[1,2], [3,4]], B = [[5,6,7], [8,9,10]], bias = [1, -1, 0.5]
    // A*B = [[21,24,27], [47,54,61]]
    // + bias = [[22,23,27.5], [48,53,61.5]]
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8, 9, 10 };
    const bias = [_]f32{ 1, -1, 0.5 };
    var C = [_]f32{ 0, 0, 0, 0, 0, 0 };

    sgemmBias(2, 3, 2, &A, &B, &bias, &C);

    try testing.expectApproxEqAbs(C[0], 22.0, 1e-5);
    try testing.expectApproxEqAbs(C[1], 23.0, 1e-5);
    try testing.expectApproxEqAbs(C[2], 27.5, 1e-5);
    try testing.expectApproxEqAbs(C[3], 48.0, 1e-5);
    try testing.expectApproxEqAbs(C[4], 53.0, 1e-5);
    try testing.expectApproxEqAbs(C[5], 61.5, 1e-5);
}

test "sgemmBias matches separate sgemm+bias" {
    // Random-ish test: compare fused vs separate for sentence-transformer-like shapes
    // M=8 (seq_len), N=384 (hidden_dim), K=384
    const M = 8;
    const N = 384;
    const K = 384;
    const allocator = testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    var bias: [N]f32 = undefined;
    const C_fused = try allocator.alloc(f32, M * N);
    defer allocator.free(C_fused);
    const C_separate = try allocator.alloc(f32, M * N);
    defer allocator.free(C_separate);

    // Fill with deterministic values
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&bias) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // Fused path
    sgemmBias(M, N, K, A, B, &bias, C_fused);

    // Separate path: sgemm then add bias
    sgemm_impl.sgemm(M, N, K, A, B, C_separate, 1.0, 0.0);
    addBiasRows(M, N, &bias, C_separate, N);

    // Compare
    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(C_fused[i], C_separate[i], 1e-4);
    }
}

test "sgemmBias skinny M=1 (single token)" {
    // M=1, N=384, K=384 — single token inference
    const N = 384;
    const K = 384;
    const allocator = testing.allocator;

    const A = try allocator.alloc(f32, K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    var bias: [N]f32 = undefined;
    const C_fused = try allocator.alloc(f32, N);
    defer allocator.free(C_fused);
    const C_separate = try allocator.alloc(f32, N);
    defer allocator.free(C_separate);

    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&bias) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    sgemmBias(1, N, K, A, B, &bias, C_fused);

    sgemm_impl.sgemm(1, N, K, A, B, C_separate, 1.0, 0.0);
    addBiasRows(1, N, &bias, C_separate, N);

    for (0..N) |i| {
        try testing.expectApproxEqAbs(C_fused[i], C_separate[i], 1e-4);
    }
}

test "sgemmBias FFN shape M=8 N=1536 K=384" {
    // FFN first linear: largest N in sentence transformer
    const M = 8;
    const N = 1536;
    const K = 384;
    const allocator = testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    const bias = try allocator.alloc(f32, N);
    defer allocator.free(bias);
    const C_fused = try allocator.alloc(f32, M * N);
    defer allocator.free(C_fused);
    const C_separate = try allocator.alloc(f32, M * N);
    defer allocator.free(C_separate);

    var rng = std.Random.DefaultPrng.init(7777);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (bias) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    sgemmBias(M, N, K, A, B, bias.ptr, C_fused);

    sgemm_impl.sgemm(M, N, K, A, B, C_separate, 1.0, 0.0);
    addBiasRows(M, N, bias.ptr, C_separate, N);

    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(C_fused[i], C_separate[i], 1e-4);
    }
}
