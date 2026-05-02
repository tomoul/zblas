// zblas/src/level2/sgemv_q8k.zig
// Q8_K Weight-Only SGEMV: y = A_q8k[M×N] * x
//
// Row-major A with per-block quantization: every BLOCK_SIZE (32) consecutive
// elements in the flat array share one f32 scale.
//
// For transformer inference: result[out_dim] = W_q8k[out_dim × in_dim] @ input[in_dim]
//
// Unlike sgemmQ8K (C = A_f32 @ dequant(B_q8k)), this operates directly on
// row-major quantized weights — no transpose needed. Each row is a separate
// dot product: y[i] = dot(dequant(A_row_i), x).

const std = @import("std");
const config = @import("../config.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);
const BLOCK_SIZE: usize = 32;
const VECS_PER_BLOCK = BLOCK_SIZE / VEC_WIDTH; // 4 on AVX2, 8 on NEON

// ============================================================================
// Public API
// ============================================================================

/// Q8_K SGEMV: y[M] = A_q8k[M×N] * x[N]
///
/// A is int8 row-major with per-block f32 scales.
/// scales[b] corresponds to A_q8k[b*32 .. (b+1)*32] in the flat array.
/// N must be a multiple of BLOCK_SIZE (32).
///
/// Parameters:
///   M: number of rows (output dimension)
///   N: number of columns (input dimension), must be multiple of 32
///   A_q8k: int8 quantized weight matrix [M × N] row-major
///   scales: per-block f32 scales, length = M * N / 32
///   x: float32 input vector [N]
///   y: float32 output vector [M]
pub fn sgemvQ8K(
    M: usize,
    N: usize,
    A_q8k: []const i8,
    scales: []const f32,
    x: []const f32,
    y: []f32,
) void {
    if (M == 0 or N == 0) return;
    std.debug.assert(A_q8k.len >= M * N);
    std.debug.assert(x.len >= N);
    std.debug.assert(y.len >= M);
    std.debug.assert(N % BLOCK_SIZE == 0); // All Qwen3.5 weight dims are 32-aligned

    const blocks_per_row = N / BLOCK_SIZE;

    // Process 4 rows at a time for instruction-level parallelism
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        sgemvQ8K4Rows(i, N, blocks_per_row, A_q8k, scales, x, y);
    }
    // Remaining rows (1-3)
    while (i < M) : (i += 1) {
        sgemvQ8K1Row(i, N, blocks_per_row, A_q8k, scales, x, y);
    }
}

// ============================================================================
// 4-row kernel: processes 4 output elements at once
// ============================================================================

fn sgemvQ8K4Rows(
    row_start: usize,
    N: usize,
    blocks_per_row: usize,
    A_q8k: []const i8,
    scales: []const f32,
    x: []const f32,
    y: []f32,
) void {
    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);

    const base0 = row_start * N;
    const base1 = (row_start + 1) * N;
    const base2 = (row_start + 2) * N;
    const base3 = (row_start + 3) * N;

    const scale_base0 = row_start * blocks_per_row;
    const scale_base1 = (row_start + 1) * blocks_per_row;
    const scale_base2 = (row_start + 2) * blocks_per_row;
    const scale_base3 = (row_start + 3) * blocks_per_row;

    for (0..blocks_per_row) |b| {
        const col_offset = b * BLOCK_SIZE;

        const s0: Vec = @splat(scales[scale_base0 + b]);
        const s1: Vec = @splat(scales[scale_base1 + b]);
        const s2: Vec = @splat(scales[scale_base2 + b]);
        const s3: Vec = @splat(scales[scale_base3 + b]);

        inline for (0..VECS_PER_BLOCK) |vi| {
            const j = col_offset + vi * VEC_WIDTH;
            const x_vec: Vec = x[j..][0..VEC_WIDTH].*;

            acc0 += dequantVec(A_q8k[base0 + j ..][0..VEC_WIDTH]) * s0 * x_vec;
            acc1 += dequantVec(A_q8k[base1 + j ..][0..VEC_WIDTH]) * s1 * x_vec;
            acc2 += dequantVec(A_q8k[base2 + j ..][0..VEC_WIDTH]) * s2 * x_vec;
            acc3 += dequantVec(A_q8k[base3 + j ..][0..VEC_WIDTH]) * s3 * x_vec;
        }
    }

    y[row_start + 0] = @reduce(.Add, acc0);
    y[row_start + 1] = @reduce(.Add, acc1);
    y[row_start + 2] = @reduce(.Add, acc2);
    y[row_start + 3] = @reduce(.Add, acc3);
}

// ============================================================================
// Single-row kernel
// ============================================================================

fn sgemvQ8K1Row(
    row: usize,
    N: usize,
    blocks_per_row: usize,
    A_q8k: []const i8,
    scales: []const f32,
    x: []const f32,
    y: []f32,
) void {
    var acc: Vec = @splat(0.0);
    const row_base = row * N;
    const scale_base = row * blocks_per_row;

    for (0..blocks_per_row) |b| {
        const col_offset = b * BLOCK_SIZE;
        const scale_vec: Vec = @splat(scales[scale_base + b]);

        inline for (0..VECS_PER_BLOCK) |vi| {
            const j = col_offset + vi * VEC_WIDTH;
            const a_vec = dequantVec(A_q8k[row_base + j ..][0..VEC_WIDTH]);
            const x_vec: Vec = x[j..][0..VEC_WIDTH].*;
            acc += (a_vec * scale_vec) * x_vec;
        }
    }

    y[row] = @reduce(.Add, acc);
}

// ============================================================================
// Helpers
// ============================================================================

/// Convert VEC_WIDTH i8 elements to f32 vector (no scaling)
inline fn dequantVec(data: *const [VEC_WIDTH]i8) Vec {
    const i8_vec: @Vector(VEC_WIDTH, i8) = data.*;
    const i32_vec: @Vector(VEC_WIDTH, i32) = i8_vec;
    return @floatFromInt(i32_vec);
}
