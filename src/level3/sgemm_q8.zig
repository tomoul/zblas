// zblas/src/level3/sgemm_q8.zig
// Q8 Weight-Only SGEMM: C = alpha * A_f32 * dequant(B_q8) + beta * C
//
// Dequantize-during-pack strategy (from design_review.md):
//   1. For large matrices: dequantize B into f32 during the packing step,
//      then reuse the existing tuned f32 cache-blocked SGEMM kernel.
//   2. For skinny-M (transformer inference): dequantize B on-the-fly in
//      a fused kernel that avoids any intermediate f32 buffer.
//
// This replaces the naive i,k,j loop in Tomoul's matmulF32Q8Simd which:
//   - Has no cache blocking (K loop re-scans entire N width of B and C)
//   - Has no packing (strided access to B rows)
//   - Is single-threaded
//
// Expected speedup: 3-10× over naive, closing the gap with PyTorch CPU.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../config.zig");
const sgemm_impl = @import("sgemm.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);

// ============================================================================
// Public API
// ============================================================================

/// Q8 SGEMM: C = A_f32[M×K] * dequant(B_q8[K×N]) (alpha=1.0, beta=0.0)
///
/// This is the fast path for inference. B is int8 with a single global scale.
/// Internally dispatches to skinny-M or dequant-then-sgemm based on M.
///
/// Parameters:
///   M, N, K: dimensions (A is M×K f32, B is K×N int8, C is M×N f32)
///   A: float32 input activations [M × K] row-major
///   B_q8: int8 quantized weights [K × N] row-major
///   scale: dequantization scale (w_f32 = w_i8 * scale)
///   C: float32 output [M × N] row-major
pub fn sgemmQ8(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_q8: []const i8,
    scale: f32,
    C: []f32,
) void {
    if (M == 0 or N == 0 or K == 0) return;
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B_q8.len >= K * N);
    std.debug.assert(C.len >= M * N);

    // Skinny-M path: fused dequant + compute, no intermediate buffer
    if (M <= config.SKINNY_M_THRESHOLD) {
        const use_skinny = (M <= 4) or (M % 4 != 0);
        if (use_skinny) {
            sgemmQ8SkinnyDispatch(M, N, K, A, K, B_q8, N, scale, C, N);
            return;
        }
    }

    // Large M: dequantize B on-the-fly during column-blocked processing
    // Uses KC-blocking to keep the dequantized f32 panel in L2 cache
    sgemmQ8Blocked(M, N, K, A, K, B_q8, N, scale, C, N);
}

/// Q8 SGEMM with general alpha/beta: C = alpha * A * dequant(B_q8) + beta * C
pub fn sgemmQ8General(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_q8: []const i8,
    scale: f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    if (M == 0 or N == 0 or K == 0) return;

    // Scale C by beta first
    if (beta == 0.0) {
        @memset(C[0 .. M * N], 0.0);
    } else if (beta != 1.0) {
        const beta_vec: Vec = @splat(beta);
        var idx: usize = 0;
        while (idx + VEC_WIDTH <= M * N) : (idx += VEC_WIDTH) {
            const c_vec: Vec = C[idx..][0..VEC_WIDTH].*;
            C[idx..][0..VEC_WIDTH].* = c_vec * beta_vec;
        }
        while (idx < M * N) : (idx += 1) {
            C[idx] *= beta;
        }
    }

    if (alpha == 0.0) return;

    if (alpha == 1.0 and beta == 0.0) {
        sgemmQ8(M, N, K, A, B_q8, scale, C);
    } else {
        // For non-unit alpha, compute into C (already scaled by beta above)
        // then we need to use accumulate mode with the alpha scaling
        sgemmQ8BlockedGeneral(M, N, K, A, K, B_q8, N, scale, C, N, alpha);
    }
}

// ============================================================================
// Skinny-M Q8 path: fused dequant-compute, zero allocation
// ============================================================================

/// Dispatch to comptime-specialized kernel for each M value
fn sgemmQ8SkinnyDispatch(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale: f32,
    C: []f32,
    ldc: usize,
) void {
    comptime var rows: usize = 1;
    inline while (rows <= config.SKINNY_M_THRESHOLD) : (rows += 1) {
        if (M == rows) {
            sgemmQ8SkinnyKernel(rows, N, K, A, lda, B_q8, ldb, scale, C, ldc);
            return;
        }
    }
}

/// Comptime-specialized skinny-M Q8 kernel
/// Dequantizes B on-the-fly: load i8 → convert to f32 → multiply → accumulate
fn sgemmQ8SkinnyKernel(
    comptime ROWS: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale: f32,
    C: []f32,
    ldc: usize,
) void {
    const scale_vec: Vec = @splat(scale);
    const VW = VEC_WIDTH;

    var j: usize = 0;

    // Tier 1: M ≤ 4 → process 3×VEC_WIDTH (24 cols) per chunk
    if (comptime ROWS <= 4) {
        const NR = 3 * VW;
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
                // Dequantize 3 vectors of B (24 int8 → 24 f32)
                const b_base = k * ldb + j;
                const b0 = dequantVec(B_q8[b_base..][0..VW], scale_vec);
                const b1 = dequantVec(B_q8[b_base + VW ..][0..VW], scale_vec);
                const b2 = dequantVec(B_q8[b_base + 2 * VW ..][0..VW], scale_vec);

                inline for (0..ROWS) |r| {
                    const a: Vec = @splat(A[r * lda + k]);
                    acc0[r] += a * b0;
                    acc1[r] += a * b1;
                    acc2[r] += a * b2;
                }
            }
            inline for (0..ROWS) |r| {
                C[r * ldc + j ..][0..VW].* = acc0[r];
                C[r * ldc + j + VW ..][0..VW].* = acc1[r];
                C[r * ldc + j + 2 * VW ..][0..VW].* = acc2[r];
            }
        }
    }

    // Tier 2: single VEC_WIDTH (8 cols) for all row counts
    while (j + VW <= N) : (j += VW) {
        var acc: [ROWS]Vec = undefined;
        inline for (0..ROWS) |r| {
            acc[r] = @splat(0.0);
        }
        for (0..K) |k| {
            const b_vec = dequantVec(B_q8[k * ldb + j ..][0..VW], scale_vec);
            inline for (0..ROWS) |r| {
                acc[r] += @as(Vec, @splat(A[r * lda + k])) * b_vec;
            }
        }
        inline for (0..ROWS) |r| {
            C[r * ldc + j ..][0..VW].* = acc[r];
        }
    }

    // Scalar tail
    while (j < N) : (j += 1) {
        var acc: [ROWS]f32 = undefined;
        inline for (0..ROWS) |r| {
            acc[r] = 0.0;
        }
        for (0..K) |k| {
            const b_val = @as(f32, @floatFromInt(B_q8[k * ldb + j])) * scale;
            inline for (0..ROWS) |r| {
                acc[r] += A[r * lda + k] * b_val;
            }
        }
        inline for (0..ROWS) |r| {
            C[r * ldc + j] = acc[r];
        }
    }
}

// ============================================================================
// Blocked Q8 path: KC-blocked dequant + f32 SGEMM kernel
// ============================================================================

/// KC-blocked Q8 SGEMM for larger M values.
/// Dequantizes B one KC-panel at a time to keep the f32 temp in L2 cache.
/// Then delegates to the existing f32 sgemm for the actual compute.
///
/// Memory: allocates KC*N f32 on the stack (up to 256*4096*4 = 4MB).
/// For very large N, falls back to full dequant + sgemm.
fn sgemmQ8Blocked(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale: f32,
    C: []f32,
    ldc: usize,
) void {
    const KC = config.KC;
    const scale_vec: Vec = @splat(scale);

    // Initialize C to zero
    for (0..M) |i| {
        @memset(C[i * ldc ..][0..N], 0.0);
    }

    // Stack buffer for one KC-panel of B dequantized to f32
    // KC=256, max N in practice ~4096 → 256*4096*4 = 4MB
    // This is large for stack; use a comptime check
    const MAX_STACK_PANEL = 256 * 4096; // 4M floats = 16MB limit
    const panel_elems = KC * N;

    if (panel_elems <= MAX_STACK_PANEL) {
        // Stack-allocated path (fast, no heap)
        sgemmQ8BlockedStack(M, N, K, A, lda, B_q8, ldb, scale_vec, scale, C, ldc, KC);
    } else {
        // Fallback: process column-by-column with skinny kernel
        // This shouldn't happen for transformer workloads (N ≤ 4096)
        sgemmQ8FallbackDirect(M, N, K, A, lda, B_q8, ldb, scale, C, ldc);
    }
}

/// Stack-allocated KC-blocked Q8 SGEMM
fn sgemmQ8BlockedStack(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale_vec: Vec,
    _: f32, // scale (unused, using scale_vec)
    C: []f32,
    ldc: usize,
    comptime KC_MAX: usize,
) void {
    const KC = KC_MAX;

    // Use threadlocal static buffer to avoid stack overflow in FFI contexts
    // KC=256, max N=4096 → 1M floats = 4MB
    const PanelBuf = struct {
        threadlocal var buf: [KC * 4096]f32 = undefined;
    };
    const b_panel: *[KC * 4096]f32 = &PanelBuf.buf;

    var k_start: usize = 0;
    while (k_start < K) : (k_start += KC) {
        const k_block = @min(KC, K - k_start);

        // Dequantize this KC-panel of B into f32
        for (0..k_block) |kk| {
            const src_row = (k_start + kk) * ldb;
            const dst_row = kk * N;
            var j: usize = 0;

            while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
                b_panel[dst_row + j ..][0..VEC_WIDTH].* = dequantVec(
                    B_q8[src_row + j ..][0..VEC_WIDTH],
                    scale_vec,
                );
            }
            // Scalar tail
            while (j < N) : (j += 1) {
                b_panel[dst_row + j] = @as(f32, @floatFromInt(B_q8[src_row + j])) * scale_vec[0];
            }
        }

        // Delegate to f32 sgemm for this block: C += A_block @ B_panel
        // A_block starts at column k_start of A (stride = lda)
        // sgemmTranspose reads A[i*lda + j] for j in 0..k_block
        // So we offset A by k_start to get element (i,j) = A[i*lda + k_start + j]
        const beta: f32 = if (k_start == 0) 0.0 else 1.0;
        sgemm_impl.sgemmTranspose(
            .NoTrans,
            .NoTrans,
            M,
            N,
            k_block,
            A[k_start..], // offset to column k_start
            lda,
            b_panel[0 .. k_block * N],
            N,
            C,
            ldc,
            1.0,
            beta,
        );
    }
}

/// Fallback direct Q8 matmul (no intermediate buffer, for very large N)
fn sgemmQ8FallbackDirect(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale: f32,
    C: []f32,
    ldc: usize,
) void {
    for (0..M) |i| {
        @memset(C[i * ldc ..][0..N], 0.0);
    }

    const scale_vec: Vec = @splat(scale);

    // Use 4-row blocking similar to sgemmDirect
    const MR = 4;
    var i: usize = 0;
    while (i + MR <= M) : (i += MR) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);

            for (0..K) |k| {
                const b_vec = dequantVec(B_q8[k * ldb + j ..][0..VEC_WIDTH], scale_vec);
                c0 += @as(Vec, @splat(A[(i + 0) * lda + k])) * b_vec;
                c1 += @as(Vec, @splat(A[(i + 1) * lda + k])) * b_vec;
                c2 += @as(Vec, @splat(A[(i + 2) * lda + k])) * b_vec;
                c3 += @as(Vec, @splat(A[(i + 3) * lda + k])) * b_vec;
            }

            C[(i + 0) * ldc + j ..][0..VEC_WIDTH].* = c0;
            C[(i + 1) * ldc + j ..][0..VEC_WIDTH].* = c1;
            C[(i + 2) * ldc + j ..][0..VEC_WIDTH].* = c2;
            C[(i + 3) * ldc + j ..][0..VEC_WIDTH].* = c3;
        }
        // scalar tail
        while (j < N) : (j += 1) {
            for (0..MR) |r| {
                var sum: f32 = 0.0;
                for (0..K) |k| {
                    sum += A[(i + r) * lda + k] * @as(f32, @floatFromInt(B_q8[k * ldb + j])) * scale;
                }
                C[(i + r) * ldc + j] = sum;
            }
        }
    }

    // Remaining rows
    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            for (0..K) |k| {
                const b_vec = dequantVec(B_q8[k * ldb + j ..][0..VEC_WIDTH], scale_vec);
                c0 += @as(Vec, @splat(A[i * lda + k])) * b_vec;
            }
            C[i * ldc + j ..][0..VEC_WIDTH].* = c0;
        }
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * @as(f32, @floatFromInt(B_q8[k * ldb + j])) * scale;
            }
            C[i * ldc + j] = sum;
        }
    }
}

/// Blocked Q8 with alpha scaling (for general case)
fn sgemmQ8BlockedGeneral(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8: []const i8,
    ldb: usize,
    scale: f32,
    C: []f32,
    ldc: usize,
    alpha: f32,
) void {
    // Simple approach: use the direct fallback with alpha
    const scale_alpha = scale * alpha;
    const scale_vec: Vec = @splat(scale_alpha);

    const MR = 4;
    var i: usize = 0;
    while (i + MR <= M) : (i += MR) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);

            for (0..K) |k| {
                const b_vec = dequantVec(B_q8[k * ldb + j ..][0..VEC_WIDTH], scale_vec);
                c0 += @as(Vec, @splat(A[(i + 0) * lda + k])) * b_vec;
                c1 += @as(Vec, @splat(A[(i + 1) * lda + k])) * b_vec;
                c2 += @as(Vec, @splat(A[(i + 2) * lda + k])) * b_vec;
                c3 += @as(Vec, @splat(A[(i + 3) * lda + k])) * b_vec;
            }

            const c0_ptr = C[(i + 0) * ldc + j ..][0..VEC_WIDTH];
            const c1_ptr = C[(i + 1) * ldc + j ..][0..VEC_WIDTH];
            const c2_ptr = C[(i + 2) * ldc + j ..][0..VEC_WIDTH];
            const c3_ptr = C[(i + 3) * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0;
            c1_ptr.* = @as(Vec, c1_ptr.*) + c1;
            c2_ptr.* = @as(Vec, c2_ptr.*) + c2;
            c3_ptr.* = @as(Vec, c3_ptr.*) + c3;
        }

        while (j < N) : (j += 1) {
            for (0..MR) |r| {
                var sum: f32 = 0.0;
                for (0..K) |k| {
                    sum += A[(i + r) * lda + k] * @as(f32, @floatFromInt(B_q8[k * ldb + j])) * scale_alpha;
                }
                C[(i + r) * ldc + j] += sum;
            }
        }
    }

    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            for (0..K) |k| {
                const b_vec = dequantVec(B_q8[k * ldb + j ..][0..VEC_WIDTH], scale_vec);
                c0 += @as(Vec, @splat(A[i * lda + k])) * b_vec;
            }
            const c0_ptr = C[i * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0;
        }
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * @as(f32, @floatFromInt(B_q8[k * ldb + j])) * scale_alpha;
            }
            C[i * ldc + j] += sum;
        }
    }
}

// ============================================================================
// Dequantization helpers
// ============================================================================

/// Dequantize VEC_WIDTH int8 values to f32: result = int8_to_f32(src) * scale
inline fn dequantVec(src: *const [VEC_WIDTH]i8, scale_vec: Vec) Vec {
    // Load int8 → widen to i16 → convert to f32 → scale
    // On x86_64 with AVX2, Zig compiles this to:
    //   vpmovsxbw (i8→i16), vpmovsxwd (i16→i32), vcvtdq2ps (i32→f32), vmulps
    const VecI8 = @Vector(VEC_WIDTH, i8);
    const VecI16 = @Vector(VEC_WIDTH, i16);

    const w_i8: VecI8 = src.*;
    const w_i16: VecI16 = w_i8;
    const w_f32: Vec = @floatFromInt(w_i16);
    return w_f32 * scale_vec;
}

// ============================================================================
// Tests
// ============================================================================

test "sgemmQ8 basic correctness" {
    const M = 4;
    const K = 8;
    const N = 16;

    // Create test data
    var A: [M * K]f32 = undefined;
    for (&A, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
    }

    var B_q8: [K * N]i8 = undefined;
    for (&B_q8, 0..) |*v, i| {
        v.* = @as(i8, @intCast(@as(i32, @intCast(i % 11)) - 5));
    }

    const scale: f32 = 0.05;

    // Compute reference result with naive method
    var C_ref: [M * N]f32 = undefined;
    @memset(&C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            for (0..N) |j| {
                C_ref[i * N + j] += A[i * K + k] * @as(f32, @floatFromInt(B_q8[k * N + j])) * scale;
            }
        }
    }

    // Compute with sgemmQ8
    var C_test: [M * N]f32 = undefined;
    sgemmQ8(M, N, K, &A, &B_q8, scale, &C_test);

    // Compare
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > 1e-4) {
            std.debug.print("Mismatch at {}: ref={d:.6} test={d:.6} diff={d:.6}\n", .{ idx, C_ref[idx], C_test[idx], diff });
            @panic("sgemmQ8 correctness failure");
        }
    }
}

test "sgemmQ8 skinny M=9 (transformer short text)" {
    const M = 9;
    const K = 1024;
    const N = 1024;

    const allocator = std.testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_q8 = try allocator.alloc(i8, K * N);
    defer allocator.free(B_q8);
    const C_test = try allocator.alloc(f32, M * N);
    defer allocator.free(C_test);
    const C_ref = try allocator.alloc(f32, M * N);
    defer allocator.free(C_ref);

    // Initialize
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_q8) |*v| v.* = @as(i8, @intCast(@as(i32, random.intRangeAtMost(i8, -127, 127))));
    const scale: f32 = 0.023;

    // Reference (naive)
    @memset(C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            for (0..N) |j| {
                C_ref[i * N + j] += a_val * @as(f32, @floatFromInt(B_q8[k * N + j])) * scale;
            }
        }
    }

    // sgemmQ8
    sgemmQ8(M, N, K, A, B_q8, scale, C_test);

    // Compare with tolerance (Q8 has quantization noise)
    var max_diff: f32 = 0.0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) max_diff = diff;
    }

    // Should match very closely since both use same scale
    try std.testing.expect(max_diff < 0.01);
}
