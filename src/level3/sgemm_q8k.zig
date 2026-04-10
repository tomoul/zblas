// zblas/src/level3/sgemm_q8k.zig
// Q8_K Weight-Only SGEMM: C = A_f32 * dequant(B_q8k)
//
// Q8_K uses block-wise quantization: every BLOCK_SIZE (32) consecutive elements
// share one f32 scale. Weights are row-major: B[k*N + j], so within a row,
// blocks are sequential 32-element chunks.
//
// Key insight: block_size=32 aligns with VECS_PER_BLOCK×VEC_WIDTH on any arch.
// Within each group of VECS_PER_BLOCK vectors, all 32 elements share one scale →
// only one scale lookup per VECS_PER_BLOCK SIMD vectors.
//   AVX2 (VW=8):  4 Vecs × 8 = 32
//   NEON (VW=4):  8 Vecs × 4 = 32
//
// Strategy:
//   - Skinny-M (M ≤ threshold): fused dequant + compute, zero allocation
//   - Large M: KC-blocked dequant B panel to f32, delegate to f32 sgemm

const std = @import("std");
const config = @import("../config.zig");
const sgemm_impl = @import("sgemm.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);
const BLOCK_SIZE: usize = 32;
const VECS_PER_BLOCK = BLOCK_SIZE / VEC_WIDTH; // 4 on AVX2, 8 on NEON

// ============================================================================
// Public API
// ============================================================================

/// Q8_K SGEMM: C = A_f32[M×K] * dequant(B_q8k[K×N])
///
/// B is int8 with per-block scales (every 32 elements share one scale).
/// scales[i] corresponds to B.data[i*32 .. (i+1)*32].
///
/// Parameters:
///   M, N, K: dimensions
///   A: float32 activations [M × K] row-major
///   B_q8k: int8 quantized weights [K × N] row-major
///   scales: per-block f32 scales, length = ceil(K*N / 32)
///   block_size: quantization block size (must be 32)
///   C: float32 output [M × N] row-major
pub fn sgemmQ8K(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_q8k: []const i8,
    scales: []const f32,
    block_size: usize,
    C: []f32,
) void {
    if (M == 0 or N == 0 or K == 0) return;
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B_q8k.len >= K * N);
    std.debug.assert(C.len >= M * N);
    std.debug.assert(block_size == BLOCK_SIZE);

    if (M <= config.SKINNY_M_THRESHOLD) {
        sgemmQ8KSkinnyDispatch(M, N, K, A, K, B_q8k, N, scales, C, N);
        return;
    }

    sgemmQ8KBlocked(M, N, K, A, K, B_q8k, N, scales, C, N);
}

// ============================================================================
// Skinny-M Q8_K path: fused dequant-compute
// ============================================================================

fn sgemmQ8KSkinnyDispatch(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8k: []const i8,
    ldb: usize,
    scales: []const f32,
    C: []f32,
    ldc: usize,
) void {
    comptime var rows: usize = 1;
    inline while (rows <= config.SKINNY_M_THRESHOLD) : (rows += 1) {
        if (M == rows) {
            sgemmQ8KSkinnyKernel(rows, N, K, A, lda, B_q8k, ldb, scales, C, ldc);
            return;
        }
    }
    unreachable;
}

/// Comptime-specialized skinny-M Q8_K kernel.
/// Processes B in groups of BLOCK_SIZE (32) columns when N is aligned,
/// using one scale per 32-element group.
///
/// Uses VECS_PER_BLOCK accumulators to cover the full BLOCK_SIZE:
///   AVX2 (VW=8):  4 Vecs × 8 = 32
///   NEON (VW=4):  8 Vecs × 4 = 32
fn sgemmQ8KSkinnyKernel(
    comptime ROWS: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8k: []const i8,
    ldb: usize,
    scales: []const f32,
    C: []f32,
    ldc: usize,
) void {
    const VW = VEC_WIDTH;

    var j: usize = 0;

    // Main loop: process BLOCK_SIZE (32) columns at a time = VECS_PER_BLOCK vectors
    // All 32 elements within a block share one scale
    while (j + BLOCK_SIZE <= N) : (j += BLOCK_SIZE) {
        // Initialize VECS_PER_BLOCK accumulators per row
        var accs: [VECS_PER_BLOCK][ROWS]Vec = undefined;
        inline for (0..VECS_PER_BLOCK) |vi| {
            inline for (0..ROWS) |r| {
                accs[vi][r] = @splat(0.0);
            }
        }

        for (0..K) |k| {
            const b_base = k * ldb + j;
            // Block index: flat index / 32
            const block_idx = b_base / BLOCK_SIZE;
            const scale_vec: Vec = @splat(scales[block_idx]);

            // Dequantize VECS_PER_BLOCK vectors (32 elements) with same scale
            inline for (0..VECS_PER_BLOCK) |vi| {
                const b = dequantVec(B_q8k[b_base + vi * VW ..][0..VW], scale_vec);
                inline for (0..ROWS) |r| {
                    accs[vi][r] += @as(Vec, @splat(A[r * lda + k])) * b;
                }
            }
        }

        inline for (0..VECS_PER_BLOCK) |vi| {
            inline for (0..ROWS) |r| {
                C[r * ldc + j + vi * VW ..][0..VW].* = accs[vi][r];
            }
        }
    }

    // Remainder: single vector at a time (need per-element block lookup)
    while (j + VW <= N) : (j += VW) {
        var acc: [ROWS]Vec = undefined;
        inline for (0..ROWS) |r| {
            acc[r] = @splat(0.0);
        }
        for (0..K) |k| {
            const b_base = k * ldb + j;
            const block_idx = b_base / BLOCK_SIZE;
            // Check if this VW straddles a block boundary
            const end_block_idx = (b_base + VW - 1) / BLOCK_SIZE;
            const scale_vec: Vec = if (block_idx == end_block_idx)
                @splat(scales[block_idx])
            else blk: {
                // Straddles boundary — build per-element scale
                var sv: Vec = undefined;
                inline for (0..VW) |vi| {
                    sv[vi] = scales[(b_base + vi) / BLOCK_SIZE];
                }
                break :blk sv;
            };

            const b_vec = dequantVecScaled(B_q8k[b_base..][0..VW], scale_vec);
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
            const w_idx = k * ldb + j;
            const scale = scales[w_idx / BLOCK_SIZE];
            const b_val = @as(f32, @floatFromInt(B_q8k[w_idx])) * scale;
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
// Blocked Q8_K path: KC-blocked dequant + f32 SGEMM kernel
// ============================================================================

/// Large-M Q8_K path: process in row chunks using the skinny kernel.
///
/// The skinny kernel uses a specific accumulation order (k=0..K-1 per output element)
/// that matches single-sentence processing. Using a different accumulation order
/// (e.g., GotoBLAS tiling via sgemmOptimized) produces tiny per-element differences
/// that compound catastrophically through deep transformer layers (6+ layers of
/// matmul→layernorm→attention→FFN), causing batch results to diverge from
/// sequential results (cosine ~0.4 instead of ~1.0).
///
/// Splitting into row chunks of ≤ SKINNY_M_THRESHOLD ensures bitwise-identical
/// per-row results regardless of total M, at the cost of re-reading B for each chunk.
/// For typical transformer dimensions (K≤1536, N≤1536), B fits in L2/L3 cache,
/// so the performance impact is modest (~1.5-2x for B reads vs a fully blocked path).
fn sgemmQ8KBlocked(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B_q8k: []const i8,
    ldb: usize,
    scales: []const f32,
    C: []f32,
    ldc: usize,
) void {
    const CHUNK = config.SKINNY_M_THRESHOLD;

    var row: usize = 0;
    while (row < M) : (row += CHUNK) {
        const chunk_m = @min(CHUNK, M - row);
        sgemmQ8KSkinnyDispatch(
            chunk_m,
            N,
            K,
            A[row * lda ..],
            lda,
            B_q8k,
            ldb,
            scales,
            C[row * ldc ..],
            ldc,
        );
    }
}

// ============================================================================
// Dequantization helpers
// ============================================================================

inline fn dequantVec(src: *const [VEC_WIDTH]i8, scale_vec: Vec) Vec {
    const VecI8 = @Vector(VEC_WIDTH, i8);
    const VecI16 = @Vector(VEC_WIDTH, i16);
    const w_i8: VecI8 = src.*;
    const w_i16: VecI16 = w_i8;
    const w_f32: Vec = @floatFromInt(w_i16);
    return w_f32 * scale_vec;
}

inline fn dequantVecScaled(src: *const [VEC_WIDTH]i8, scale_vec: Vec) Vec {
    return dequantVec(src, scale_vec);
}

// ============================================================================
// Tests
// ============================================================================

test "sgemmQ8K basic correctness" {
    const M = 4;
    const K = 8;
    const N = 32; // Must be multiple of BLOCK_SIZE for simple test

    var A: [M * K]f32 = undefined;
    for (&A, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
    }

    var B_q8k: [K * N]i8 = undefined;
    for (&B_q8k, 0..) |*v, i| {
        v.* = @as(i8, @intCast(@as(i32, @intCast(i % 11)) - 5));
    }

    // One scale per block of 32 elements
    const num_blocks = (K * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    var block_scales: [num_blocks]f32 = undefined;
    for (&block_scales, 0..) |*s, i| {
        s.* = 0.03 + @as(f32, @floatFromInt(i % 5)) * 0.01;
    }

    // Reference computation
    var C_ref: [M * N]f32 = undefined;
    @memset(&C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            for (0..N) |j| {
                const w_idx = k * N + j;
                const scale = block_scales[w_idx / BLOCK_SIZE];
                C_ref[i * N + j] += A[i * K + k] * @as(f32, @floatFromInt(B_q8k[w_idx])) * scale;
            }
        }
    }

    var C_test: [M * N]f32 = undefined;
    sgemmQ8K(M, N, K, &A, &B_q8k, &block_scales, BLOCK_SIZE, &C_test);

    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > 1e-4) {
            std.debug.print("Mismatch at {}: ref={d:.6} test={d:.6} diff={d:.6}\n", .{ idx, C_ref[idx], C_test[idx], diff });
            @panic("sgemmQ8K correctness failure");
        }
    }
}

test "sgemmQ8K sentence transformer sizes (M=11, N=384, K=384)" {
    const M = 11;
    const K = 384;
    const N = 384;

    const allocator = std.testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_q8k = try allocator.alloc(i8, K * N);
    defer allocator.free(B_q8k);
    const num_blocks = (K * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const block_scales = try allocator.alloc(f32, num_blocks);
    defer allocator.free(block_scales);
    const C_test = try allocator.alloc(f32, M * N);
    defer allocator.free(C_test);
    const C_ref = try allocator.alloc(f32, M * N);
    defer allocator.free(C_ref);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_q8k) |*v| v.* = @as(i8, @intCast(@as(i32, random.intRangeAtMost(i8, -127, 127))));
    for (block_scales) |*s| s.* = random.float(f32) * 0.1;

    // Reference
    @memset(C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            for (0..N) |j| {
                const w_idx = k * N + j;
                C_ref[i * N + j] += a_val * @as(f32, @floatFromInt(B_q8k[w_idx])) * block_scales[w_idx / BLOCK_SIZE];
            }
        }
    }

    sgemmQ8K(M, N, K, A, B_q8k, block_scales, BLOCK_SIZE, C_test);

    var max_diff: f32 = 0.0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) max_diff = diff;
    }

    try std.testing.expect(max_diff < 0.01);
}

test "sgemmQ8K blocked path (M=50 > SKINNY_M_THRESHOLD)" {
    const M = 50; // Triggers blocked path (> SKINNY_M_THRESHOLD=32)
    const K = 384;
    const N = 384;

    const allocator = std.testing.allocator;

    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_q8k = try allocator.alloc(i8, K * N);
    defer allocator.free(B_q8k);
    const num_blocks = (K * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const block_scales = try allocator.alloc(f32, num_blocks);
    defer allocator.free(block_scales);
    const C_test = try allocator.alloc(f32, M * N);
    defer allocator.free(C_test);
    const C_ref = try allocator.alloc(f32, M * N);
    defer allocator.free(C_ref);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_q8k) |*v| v.* = @as(i8, @intCast(@as(i32, random.intRangeAtMost(i8, -127, 127))));
    for (block_scales) |*s| s.* = random.float(f32) * 0.1;

    // Reference: naive dequant + matmul
    @memset(C_ref, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            for (0..N) |j| {
                const w_idx = k * N + j;
                C_ref[i * N + j] += a_val * @as(f32, @floatFromInt(B_q8k[w_idx])) * block_scales[w_idx / BLOCK_SIZE];
            }
        }
    }

    sgemmQ8K(M, N, K, A, B_q8k, block_scales, BLOCK_SIZE, C_test);

    var max_diff: f32 = 0.0;
    var max_idx: usize = 0;
    for (0..M * N) |idx| {
        const diff = @abs(C_ref[idx] - C_test[idx]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = idx;
        }
    }

    if (max_diff >= 0.01) {
        std.debug.print("BLOCKED PATH FAILURE: max_diff={d:.6} at idx={} (row={}, col={})\n", .{
            max_diff, max_idx, max_idx / N, max_idx % N,
        });
        std.debug.print("  ref={d:.6} test={d:.6}\n", .{ C_ref[max_idx], C_test[max_idx] });
    }

    try std.testing.expect(max_diff < 0.01);
}
