// zblas/src/level3/sgemm_f16.zig
// F16 Weight-Only SGEMM: C = alpha * A_f32 * cast(B_f16) + beta * C
//
// Half-precision weight storage with full-precision compute:
//   1. For skinny-M (M ≤ 32, transformer inference): load f16 weights,
//      convert to f32 on-the-fly in a fused kernel. No intermediate buffer.
//   2. For large M: convert B_f16 into f32 during KC-blocked processing,
//      then reuse the existing tuned f32 cache-blocked SGEMM kernel.
//
// Benefits over f32 weights:
//   - 2× smaller model files (f16 = 2 bytes vs f32 = 4 bytes)
//   - 2× less memory bandwidth for loading B
//   - ≤5% latency overhead (f16→f32 conversion is ~1 cycle via VCVTPH2PS)
//
// On x86_64 with F16C extension (Ivy Bridge+):
//   @floatCast(@Vector(8, f16)) compiles to VCVTPH2PS (8-wide f16→f32)

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../config.zig");
const sgemm_impl = @import("sgemm.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);
const VecF16 = @Vector(VEC_WIDTH, f16);

// ============================================================================
// Public API
// ============================================================================

/// F16 SGEMM: C = A_f32[M×K] * cast(B_f16[K×N]) (alpha=1.0, beta=0.0)
///
/// Fast path for inference with half-precision weights. B is stored as f16
/// and converted to f32 on-the-fly during computation.
/// Dispatches to skinny-M or blocked path based on M.
///
/// Parameters:
///   M, N, K: dimensions (A is M×K f32, B is K×N f16, C is M×N f32)
///   A: float32 input activations [M × K] row-major
///   B_f16: float16 weights [K × N] row-major
///   C: float32 output [M × N] row-major
pub fn sgemmF16(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_f16: []const f16,
    C: []f32,
) void {
    if (M == 0 or N == 0 or K == 0) return;
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B_f16.len >= K * N);
    std.debug.assert(C.len >= M * N);

    // Skinny-M path: fused convert + compute, no intermediate buffer
    if (M <= config.SKINNY_M_THRESHOLD) {
        const use_skinny = (M <= 4) or (M % 4 != 0);
        if (use_skinny) {
            sgemmF16SkinnyDispatch(M, N, K, A, K, B_f16, N, C, N);
            return;
        }
    }

    // Large M: convert B_f16 on-the-fly during column-blocked processing
    sgemmF16Blocked(M, N, K, A, K, B_f16, N, C, N);
}

/// F16 SGEMM with general alpha/beta: C = alpha * A * cast(B_f16) + beta * C
pub fn sgemmF16General(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_f16: []const f16,
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
        sgemmF16(M, N, K, A, B_f16, C);
    } else {
        // For non-unit alpha, compute with alpha scaling
        sgemmF16BlockedGeneral(M, N, K, A, K, B_f16, N, C, N, alpha);
    }
}

// ============================================================================
// Skinny-M F16 path: fused convert-compute, zero allocation
// ============================================================================

/// Dispatch to comptime-specialized kernel for each M value
fn sgemmF16SkinnyDispatch(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
) void {
    comptime var rows: usize = 1;
    inline while (rows <= config.SKINNY_M_THRESHOLD) : (rows += 1) {
        if (M == rows) {
            sgemmF16SkinnyKernel(rows, N, K, A, lda, B_f16, ldb, C, ldc);
            return;
        }
    }
}

/// Comptime-specialized skinny-M F16 kernel
/// Loads f16 weights, converts to f32 on-the-fly, accumulates into f32
fn sgemmF16SkinnyKernel(
    comptime ROWS: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
) void {
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
                const b_base = k * ldb + j;
                // Load f16 vectors, convert to f32 (VCVTPH2PS on x86_64)
                const b0: Vec = @floatCast(@as(VecF16, B_f16[b_base..][0..VW].*));
                const b1: Vec = @floatCast(@as(VecF16, B_f16[b_base + VW ..][0..VW].*));
                const b2: Vec = @floatCast(@as(VecF16, B_f16[b_base + 2 * VW ..][0..VW].*));

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
            const b_vec: Vec = @floatCast(@as(VecF16, B_f16[k * ldb + j ..][0..VW].*));
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
            const b_val: f32 = @floatCast(B_f16[k * ldb + j]);
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
// Blocked F16 path: KC-blocked convert + f32 SGEMM kernel
// ============================================================================

/// KC-blocked F16 SGEMM for larger M values.
/// Converts B_f16 one KC-panel at a time to keep the f32 temp in L2 cache.
/// Then delegates to the existing f32 sgemm for the actual compute.
fn sgemmF16Blocked(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
) void {
    const KC = config.KC;

    // Initialize C to zero
    for (0..M) |i| {
        @memset(C[i * ldc ..][0..N], 0.0);
    }

    // Stack buffer for one KC-panel of B converted to f32
    const MAX_STACK_PANEL = 256 * 4096; // 4M floats = 16MB limit
    const panel_elems = KC * N;

    if (panel_elems <= MAX_STACK_PANEL) {
        sgemmF16BlockedStack(M, N, K, A, lda, B_f16, ldb, C, ldc, KC);
    } else {
        // Fallback: direct compute for very large N (shouldn't happen for transformers)
        sgemmF16FallbackDirect(M, N, K, A, lda, B_f16, ldb, C, ldc);
    }
}

/// Stack-allocated KC-blocked F16 SGEMM
fn sgemmF16BlockedStack(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
    comptime KC_MAX: usize,
) void {
    const KC = KC_MAX;
    const VW = VEC_WIDTH;

    // Use threadlocal static buffer to avoid stack overflow in FFI contexts
    const PanelBuf = struct {
        threadlocal var buf: [KC * 4096]f32 = undefined;
    };
    const b_panel: *[KC * 4096]f32 = &PanelBuf.buf;

    var k_start: usize = 0;
    while (k_start < K) : (k_start += KC) {
        const k_block = @min(KC, K - k_start);

        // Convert this KC-panel of B_f16 into f32
        for (0..k_block) |kk| {
            const src_row = (k_start + kk) * ldb;
            const dst_row = kk * N;
            var j: usize = 0;

            // SIMD: load VW f16 elements, convert to f32
            while (j + VW <= N) : (j += VW) {
                const b_f16_vec: VecF16 = B_f16[src_row + j ..][0..VW].*;
                const b_f32_vec: Vec = @floatCast(b_f16_vec);
                b_panel[dst_row + j ..][0..VW].* = b_f32_vec;
            }
            // Scalar tail
            while (j < N) : (j += 1) {
                b_panel[dst_row + j] = @as(f32, @floatCast(B_f16[src_row + j]));
            }
        }

        // Delegate to f32 sgemm for this block: C += A_block @ B_panel
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

/// Fallback direct F16 matmul (no intermediate buffer, for very large N)
fn sgemmF16FallbackDirect(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
) void {
    for (0..M) |i| {
        @memset(C[i * ldc ..][0..N], 0.0);
    }

    // Use 4-row blocking
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
                const b_vec: Vec = @floatCast(@as(VecF16, B_f16[k * ldb + j ..][0..VEC_WIDTH].*));
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
        // Scalar tail
        while (j < N) : (j += 1) {
            for (0..MR) |r| {
                var sum: f32 = 0.0;
                for (0..K) |k| {
                    sum += A[(i + r) * lda + k] * @as(f32, @floatCast(B_f16[k * ldb + j]));
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
                const b_vec: Vec = @floatCast(@as(VecF16, B_f16[k * ldb + j ..][0..VEC_WIDTH].*));
                c0 += @as(Vec, @splat(A[i * lda + k])) * b_vec;
            }
            C[i * ldc + j ..][0..VEC_WIDTH].* = c0;
        }
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * @as(f32, @floatCast(B_f16[k * ldb + j]));
            }
            C[i * ldc + j] = sum;
        }
    }
}

/// Blocked F16 with alpha scaling (for general case)
fn sgemmF16BlockedGeneral(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_f16: []const f16,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
) void {
    const alpha_vec: Vec = @splat(alpha);

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
                const b_vec: Vec = @floatCast(@as(VecF16, B_f16[k * ldb + j ..][0..VEC_WIDTH].*));
                c0 += @as(Vec, @splat(A[(i + 0) * lda + k])) * b_vec;
                c1 += @as(Vec, @splat(A[(i + 1) * lda + k])) * b_vec;
                c2 += @as(Vec, @splat(A[(i + 2) * lda + k])) * b_vec;
                c3 += @as(Vec, @splat(A[(i + 3) * lda + k])) * b_vec;
            }

            const c0_ptr = C[(i + 0) * ldc + j ..][0..VEC_WIDTH];
            const c1_ptr = C[(i + 1) * ldc + j ..][0..VEC_WIDTH];
            const c2_ptr = C[(i + 2) * ldc + j ..][0..VEC_WIDTH];
            const c3_ptr = C[(i + 3) * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0 * alpha_vec;
            c1_ptr.* = @as(Vec, c1_ptr.*) + c1 * alpha_vec;
            c2_ptr.* = @as(Vec, c2_ptr.*) + c2 * alpha_vec;
            c3_ptr.* = @as(Vec, c3_ptr.*) + c3 * alpha_vec;
        }

        while (j < N) : (j += 1) {
            for (0..MR) |r| {
                var sum: f32 = 0.0;
                for (0..K) |k| {
                    sum += A[(i + r) * lda + k] * @as(f32, @floatCast(B_f16[k * ldb + j]));
                }
                C[(i + r) * ldc + j] += sum * alpha;
            }
        }
    }

    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            for (0..K) |k| {
                const b_vec: Vec = @floatCast(@as(VecF16, B_f16[k * ldb + j ..][0..VEC_WIDTH].*));
                c0 += @as(Vec, @splat(A[i * lda + k])) * b_vec;
            }
            const c0_ptr = C[i * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0 * alpha_vec;
        }
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * @as(f32, @floatCast(B_f16[k * ldb + j]));
            }
            C[i * ldc + j] += sum * alpha;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sgemmF16 basic correctness 4x8x16" {
    const M = 4;
    const K = 8;
    const N = 16;

    // Create test data
    var A: [M * K]f32 = undefined;
    for (&A, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
    }

    var B_f16: [K * N]f16 = undefined;
    for (&B_f16, 0..) |*v, i| {
        v.* = @as(f16, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * @as(f16, 0.05);
    }

    // Compute reference result with naive method (convert B to f32 first)
    var C_ref: [M * N]f32 = undefined;
    @memset(&C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            for (0..N) |j| {
                C_ref[i * N + j] += A[i * K + k] * @as(f32, @floatCast(B_f16[k * N + j]));
            }
        }
    }

    // Compute with sgemmF16
    var C_test: [M * N]f32 = undefined;
    sgemmF16(M, N, K, &A, &B_f16, &C_test);

    // Compare (f16 has ~3 decimal digits precision)
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > 1e-3) {
            std.debug.print("Mismatch at {}: ref={d:.6} test={d:.6} diff={d:.6}\n", .{ idx, C_ref[idx], C_test[idx], diff });
            @panic("sgemmF16 correctness failure");
        }
    }
}

test "sgemmF16 skinny M=1 (single vector)" {
    const M = 1;
    const K = 32;
    const N = 64;

    var A: [M * K]f32 = undefined;
    var B_f16: [K * N]f16 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&B_f16) |*v| v.* = @floatCast(random.float(f32) * 2.0 - 1.0);

    // Reference
    var C_ref: [M * N]f32 = undefined;
    @memset(&C_ref, 0.0);
    for (0..K) |k| {
        for (0..N) |j| {
            C_ref[j] += A[k] * @as(f32, @floatCast(B_f16[k * N + j]));
        }
    }

    var C_test: [M * N]f32 = undefined;
    sgemmF16(M, N, K, &A, &B_f16, &C_test);

    var max_diff: f32 = 0.0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) max_diff = diff;
    }
    // f16 accumulation should be very close (both paths use f32 compute)
    try std.testing.expect(max_diff < 0.01);
}

test "sgemmF16 skinny M=9 (transformer short text)" {
    const M = 9;
    const K = 384;
    const N = 384;

    const allocator = std.testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_f16 = try allocator.alloc(f16, K * N);
    defer allocator.free(B_f16);
    const C_test = try allocator.alloc(f32, M * N);
    defer allocator.free(C_test);
    const C_ref = try allocator.alloc(f32, M * N);
    defer allocator.free(C_ref);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_f16) |*v| v.* = @floatCast(random.float(f32) * 2.0 - 1.0);

    // Reference (naive)
    @memset(C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            for (0..N) |j| {
                C_ref[i * N + j] += a_val * @as(f32, @floatCast(B_f16[k * N + j]));
            }
        }
    }

    sgemmF16(M, N, K, A, B_f16, C_test);

    var max_diff: f32 = 0.0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) max_diff = diff;
    }

    // Should match very closely since both use same f16→f32 conversion
    try std.testing.expect(max_diff < 0.01);
}

test "sgemmF16 general alpha/beta" {
    const M = 4;
    const K = 8;
    const N = 16;

    var A: [M * K]f32 = undefined;
    var B_f16: [K * N]f16 = undefined;
    var C_test: [M * N]f32 = undefined;
    var C_ref: [M * N]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();
    for (&A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&B_f16) |*v| v.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    for (&C_test) |*v| v.* = random.float(f32);
    @memcpy(&C_ref, &C_test);

    const alpha: f32 = 2.5;
    const beta: f32 = 0.3;

    // Reference: C_ref = alpha * A @ B_f32 + beta * C_ref
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * K + k] * @as(f32, @floatCast(B_f16[k * N + j]));
            }
            C_ref[i * N + j] = alpha * sum + beta * C_ref[i * N + j];
        }
    }

    sgemmF16General(M, N, K, &A, &B_f16, &C_test, alpha, beta);

    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > 1e-3) {
            std.debug.print("Mismatch at {}: ref={d:.6} test={d:.6} diff={d:.6}\n", .{ idx, C_ref[idx], C_test[idx], diff });
            @panic("sgemmF16General correctness failure");
        }
    }
}

test "sgemmF16 large M blocked path" {
    // M > SKINNY_M_THRESHOLD to exercise the blocked path
    const M = 64;
    const K = 128;
    const N = 256;

    const allocator = std.testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_f16 = try allocator.alloc(f16, K * N);
    defer allocator.free(B_f16);
    const C_test = try allocator.alloc(f32, M * N);
    defer allocator.free(C_test);
    const C_ref = try allocator.alloc(f32, M * N);
    defer allocator.free(C_ref);

    var rng = std.Random.DefaultPrng.init(99);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_f16) |*v| v.* = @floatCast(random.float(f32) * 2.0 - 1.0);

    // Reference
    @memset(C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            for (0..N) |j| {
                C_ref[i * N + j] += a_val * @as(f32, @floatCast(B_f16[k * N + j]));
            }
        }
    }

    sgemmF16(M, N, K, A, B_f16, C_test);

    var max_diff: f32 = 0.0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) max_diff = diff;
    }

    // Blocked path may accumulate slightly more error due to KC blocking
    try std.testing.expect(max_diff < 0.05);
}
