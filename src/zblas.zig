// zblas - Pure Zig BLAS Library
//
// A drop-in replacement for CBLAS focused on AI inference workloads.
// Provides SGEMM, SGEMV, and essential Level 1 operations.
//
// Usage:
//   const zblas = @import("zblas");
//   zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);
//
// This module is API-compatible with tomoul/src/core/blas.zig

const std = @import("std");
const builtin = @import("builtin");

pub const config = @import("config.zig");
pub const reference = @import("reference.zig");
pub const packing = @import("util/packing.zig");

// Import optimized implementations
const sgemm_impl = @import("level3/sgemm.zig");
const sgemm_q8_impl = @import("level3/sgemm_q8.zig");
const sgemm_q8k_impl = @import("level3/sgemm_q8k.zig");
const sgemm_f16_impl = @import("level3/sgemm_f16.zig");
const sgemm_bias_impl = @import("level3/sgemm_bias.zig");
const sgemm_parallel_impl = @import("level3/sgemm_parallel.zig");
const sgemv_impl = @import("level2/sgemv.zig");
const sgemv_q8k_impl = @import("level2/sgemv_q8k.zig");
const level1 = @import("level1/blas_level1.zig");
const conv1d_impl = @import("conv/conv1d.zig");

// ============================================================================
// CBLAS-compatible Types
// ============================================================================

/// Matrix storage order
pub const Order = enum(c_int) {
    RowMajor = 101,
    ColMajor = 102,
};

/// Matrix transpose option
pub const Transpose = enum(c_int) {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
};

// ============================================================================
// Level 3 BLAS (Matrix-Matrix Operations)
// ============================================================================

/// Single-precision General Matrix Multiply: C = alpha*A*B + beta*C
///
/// Parameters:
///   - M: rows of A and C
///   - N: cols of B and C
///   - K: cols of A, rows of B
///   - A: matrix [M x K] in row-major order
///   - B: matrix [K x N] in row-major order
///   - C: output matrix [M x N] in row-major order
///   - alpha: scalar multiplier for A*B (typically 1.0)
///   - beta: scalar multiplier for C (0.0 to overwrite, 1.0 to accumulate)
pub inline fn sgemm(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemm_impl.sgemm(M, N, K, A, B, C, alpha, beta);
}

/// Single-precision General Matrix Multiply with transpose options
/// C = alpha * op(A) * op(B) + beta * C
///
/// Parameters:
///   - transA: whether to transpose A
///   - transB: whether to transpose B
///   - M: rows of op(A) and C
///   - N: cols of op(B) and C
///   - K: cols of op(A), rows of op(B)
///   - A: matrix (transposed if transA == .Trans)
///   - lda: leading dimension of A
///   - B: matrix (transposed if transB == .Trans)
///   - ldb: leading dimension of B
///   - C: output matrix [M x N]
///   - ldc: leading dimension of C
///   - alpha, beta: scalar multipliers
pub fn sgemmTranspose(
    transA: Transpose,
    transB: Transpose,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    sgemm_impl.sgemmTranspose(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}

// ============================================================================
// Fused SGEMM + Bias (Phase 14)
// ============================================================================

/// Fused SGEMM + bias: C = A*B + broadcast(bias)
///
/// Eliminates the separate bias addition pass by fusing it into the SGEMM
/// store epilogue. bias is a [N] vector broadcast to every row of C.
///
/// Uses specialized skinny-M kernel for M ≤ 32 (typical for inference).
/// Falls back to sgemm + separate bias add for larger M.
pub fn sgemmBias(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    bias: [*]const f32,
    C: []f32,
) void {
    sgemm_bias_impl.sgemmBias(M, N, K, A, B, bias, C);
}

// ============================================================================
// Fused SGEMM (Phase 10)
// ============================================================================

/// Fused SGEMM: C = alpha*A*B + beta*C with fused B packing
///
/// This variant eliminates the separate B packing step by reading B directly
/// during computation. Benefits:
/// - Reduces memory traffic by ~33% (no packed B buffer writes/reads)
/// - Smaller working set (no KC*NC packed B buffer needed)
/// - Can be faster for memory-bound workloads
///
/// Trade-offs:
/// - B access pattern is less regular than packed format
/// - May be slower if B is not cache-resident
/// - Only works for NN (no transpose) case
///
/// Use this when:
/// - Matrix is large enough that packing overhead matters (>512 dimensions)
/// - Memory bandwidth is the bottleneck
/// - B is not reused across multiple operations
///
/// Parameters:
///   - M, N, K: Matrix dimensions (A is MxK, B is KxN, C is MxN)
///   - A, B: Input matrices in row-major order
///   - C: Output matrix in row-major order
///   - alpha, beta: Scalar multipliers
pub fn sgemmFused(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemm_impl.sgemmFused(M, N, K, A, K, B, N, C, N, alpha, beta);
}

/// Fused SGEMM with explicit leading dimensions
///
/// Same as sgemmFused but allows custom leading dimensions for submatrix operations.
pub fn sgemmFusedLd(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    sgemm_impl.sgemmFused(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}

// ============================================================================
// Parallel SGEMM (Phase 6)
// ============================================================================

/// Parallel SGEMM with automatic thread count selection.
///
/// Automatically determines optimal thread count based on problem size.
/// Falls back to single-threaded for small matrices where threading hurts.
///
/// IMPORTANT: Based on Phase 5 benchmarks, multi-threading only helps for
/// large matrices (>50M FLOPS, >512 rows). For typical AI inference with
/// many small matmuls, single-threaded is often faster!
///
/// Parameters:
///   - allocator: Memory allocator for packing buffers
///   - M, N, K: Matrix dimensions (A is MxK, B is KxN, C is MxN)
///   - A, B: Input matrices in row-major order
///   - C: Output matrix in row-major order
///   - alpha, beta: Scalar multipliers
pub fn sgemmParallel(
    allocator: std.mem.Allocator,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) !void {
    try sgemm_parallel_impl.sgemmParallelAuto(allocator, M, N, K, A, B, C, alpha, beta);
}

/// Parallel SGEMM with explicit thread count.
///
/// Use this when you want to control threading explicitly.
/// Thread count will be automatically reduced if the problem is too small.
pub fn sgemmParallelN(
    allocator: std.mem.Allocator,
    num_threads: usize,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) !void {
    try sgemm_parallel_impl.sgemmParallel(allocator, num_threads, M, N, K, A, K, B, N, C, N, alpha, beta);
}

/// Check if a matrix size should use parallel execution.
/// Returns false for small/medium matrices where threading hurts performance.
pub fn shouldParallelize(M: usize, N: usize, K: usize) bool {
    return sgemm_parallel_impl.shouldParallelize(M, N, K);
}

/// Get optimal thread count for given matrix dimensions.
/// Returns 1 for small matrices, more for large matrices.
pub fn getOptimalThreadCount(M: usize, N: usize, K: usize) usize {
    return sgemm_parallel_impl.getOptimalThreadCount(M, N, K);
}

// ============================================================================
// Context-based API (Tomoul integration)
// ============================================================================

/// Context for zblas operations with threading configuration.
/// Compatible with Tomoul's Context struct.
pub const Context = struct {
    num_threads: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) Context {
        return .{
            .num_threads = num_threads,
            .allocator = allocator,
        };
    }

    /// SGEMM using this context's threading configuration.
    /// Automatically uses parallel path for large matrices if num_threads > 1.
    pub fn sgemm(
        ctx: *const Context,
        M: usize,
        N: usize,
        K: usize,
        A: []const f32,
        B: []const f32,
        C: []f32,
        alpha: f32,
        beta: f32,
    ) !void {
        if (ctx.num_threads > 1 and sgemm_parallel_impl.shouldParallelize(M, N, K)) {
            try sgemm_parallel_impl.sgemmParallel(
                ctx.allocator,
                ctx.num_threads,
                M,
                N,
                K,
                A,
                K,
                B,
                N,
                C,
                N,
                alpha,
                beta,
            );
        } else {
            sgemm_impl.sgemm(M, N, K, A, B, C, alpha, beta);
        }
    }
};

// ============================================================================
// Level 2 BLAS (Matrix-Vector Operations)
// ============================================================================

/// Single-precision Matrix-Vector multiply: y = alpha*A*x + beta*y
///
/// Parameters:
///   - M: rows of A
///   - N: cols of A
///   - A: matrix [M x N] in row-major order
///   - x: input vector of length N
///   - y: output vector of length M
///   - alpha, beta: scalar multipliers
pub fn sgemv(
    M: usize,
    N: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemv_impl.sgemv(M, N, A, x, y, alpha, beta);
}

/// Single-precision Matrix-Vector multiply with transpose: y = alpha*op(A)*x + beta*y
///
/// When trans == .NoTrans: A is [M×N], x is [N], y is [M]
/// When trans == .Trans:   A is [M×N], x is [M], y is [N]
pub fn sgemvTrans(
    trans: Transpose,
    M: usize,
    N: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemv_impl.sgemvTrans(trans, M, N, A, x, y, alpha, beta);
}

// ============================================================================
// Q8 Weight-Only SGEMM (Quantized Inference)
// ============================================================================

/// Q8 SGEMM: C = A_f32[M×K] * dequant(B_q8[K×N])
///
/// Weight-only quantization: float32 activations × int8 weights.
/// Dequantizes on-the-fly during computation for optimal cache usage.
/// Uses skinny-M kernel for small M (transformer inference) and
/// KC-blocked dequant for larger M.
///
/// Parameters:
///   - M, N, K: dimensions (A is M×K, B is K×N, C is M×N)
///   - A: float32 input activations
///   - B_q8: int8 quantized weights (row-major)
///   - scale: dequantization scale (w_f32 = w_i8 * scale)
///   - C: float32 output
pub fn sgemmQ8(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_q8: []const i8,
    scale: f32,
    C: []f32,
) void {
    sgemm_q8_impl.sgemmQ8(M, N, K, A, B_q8, scale, C);
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
    sgemm_q8_impl.sgemmQ8General(M, N, K, A, B_q8, scale, C, alpha, beta);
}

// ============================================================================
// Q8_K Weight-Only SGEMM (block-wise quantization, per-block scales)
// ============================================================================

/// Q8_K SGEMM: C = A_f32[M×K] * dequant(B_q8k[K×N])
///
/// B is int8 with per-block scales. Every `block_size` consecutive elements
/// in the flat B array share one f32 scale.
///
/// Parameters:
///   - M, N, K: dimensions (A is M×K, B is K×N, C is M×N)
///   - A: float32 input activations
///   - B_q8k: int8 quantized weights (row-major)
///   - scales: per-block f32 scales, one per `block_size` elements
///   - block_size: quantization block size (must be 32)
///   - C: float32 output
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
    sgemm_q8k_impl.sgemmQ8K(M, N, K, A, B_q8k, scales, block_size, C);
}

// ============================================================================
// Q8_K Weight-Only SGEMV (Row-Major, Per-Block Scales)
// ============================================================================

/// Q8_K SGEMV: y[M] = A_q8k[M×N] * x[N]
///
/// Row-major A with per-block f32 scales. Every 32 consecutive elements
/// in the flat A array share one f32 scale. N must be a multiple of 32.
///
/// For transformer inference: result[out_dim] = W_q8k[out_dim × in_dim] @ input[in_dim]
///
/// Parameters:
///   - M: number of rows (output dimension)
///   - N: number of columns (input dimension), must be multiple of 32
///   - A_q8k: int8 quantized weight matrix (row-major)
///   - scales: per-block f32 scales, one per 32 elements
///   - x: float32 input vector
///   - y: float32 output vector
pub fn sgemvQ8K(
    M: usize,
    N: usize,
    A_q8k: []const i8,
    scales: []const f32,
    x: []const f32,
    y: []f32,
) void {
    sgemv_q8k_impl.sgemvQ8K(M, N, A_q8k, scales, x, y);
}

// ============================================================================
// F16 Weight-Only SGEMM (Half-Precision Weights, Phase 18)
// ============================================================================

/// F16 SGEMM: C = A_f32[M×K] * cast(B_f16[K×N])
///
/// Half-precision weight storage with full-precision compute.
/// Weights stored as f16 (2× smaller than f32) and converted to f32
/// on-the-fly during computation. Uses skinny-M kernel for small M
/// (transformer inference) and KC-blocked convert for larger M.
///
/// Parameters:
///   - M, N, K: dimensions (A is M×K, B is K×N, C is M×N)
///   - A: float32 input activations
///   - B_f16: float16 weights (row-major)
///   - C: float32 output
pub fn sgemmF16(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B_f16: []const f16,
    C: []f32,
) void {
    sgemm_f16_impl.sgemmF16(M, N, K, A, B_f16, C);
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
    sgemm_f16_impl.sgemmF16General(M, N, K, A, B_f16, C, alpha, beta);
}

// ============================================================================
// Level 1 BLAS (Vector Operations)
// ============================================================================

/// SAXPY: y = alpha * x + y
pub fn saxpy(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    level1.saxpy(n, alpha, x, y);
}

/// SDOT: dot product of two vectors
pub fn sdot(n: usize, x: []const f32, y: []const f32) f32 {
    return level1.sdot(n, x, y);
}

/// SSCAL: x = alpha * x
pub fn sscal(n: usize, alpha: f32, x: []f32) void {
    level1.sscal(n, alpha, x);
}

/// SNRM2: Euclidean norm ||x||_2
pub fn snrm2(n: usize, x: []const f32) f32 {
    return level1.snrm2(n, x);
}

/// SCOPY: y = x
pub fn scopy(n: usize, x: []const f32, y: []f32) void {
    level1.scopy(n, x, y);
}

/// ISAMAX: index of element with max absolute value
pub fn isamax(n: usize, x: []const f32) usize {
    return level1.isamax(n, x);
}

// ============================================================================
// Conv1d (SIMD-optimized 1D Convolution)
// ============================================================================

/// Activation function for fused conv1d epilogue.
pub const Activation = conv1d_impl.Activation;

/// Repack convolution weights from standard [C_out, C_in, K] layout to
/// SIMD-friendly [C_in, K, C_out] layout for use with conv1d/conv1dNoPad.
pub fn conv1dRepackWeight(
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
    src: [*]const f32,
    dst: [*]f32,
) void {
    conv1d_impl.repackWeight(out_channels, in_channels, kernel_size, src, dst);
}

/// SIMD Conv1d with optional fused bias and activation.
///
/// Weight must be repacked via conv1dRepackWeight first.
/// Handles arbitrary padding with bounds-checked inner loop.
pub inline fn conv1dFull(
    out_channels: usize,
    in_channels: usize,
    out_width: usize,
    in_width: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
    comptime activation: Activation,
) void {
    conv1d_impl.conv1d(out_channels, in_channels, out_width, in_width, kernel_size, stride, padding, input, weight, bias, output, activation);
}

/// Optimized Conv1d for padding=0 (no bounds checks in inner loop).
///
/// Weight must be repacked via conv1dRepackWeight first.
pub inline fn conv1dNoPad(
    out_channels: usize,
    in_channels: usize,
    out_width: usize,
    in_width: usize,
    kernel_size: usize,
    stride: usize,
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
    comptime activation: Activation,
) void {
    conv1d_impl.conv1dNoPad(out_channels, in_channels, out_width, in_width, kernel_size, stride, input, weight, bias, output, activation);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get configuration information
pub fn getInfo() struct {
    arch: []const u8,
    simd: bool,
    vector_width: usize,
    mr: usize,
    nr: usize,
} {
    return .{
        .arch = @tagName(builtin.cpu.arch),
        .simd = config.hasSimd(),
        .vector_width = config.getVectorWidth(),
        .mr = config.MR,
        .nr = config.NR,
    };
}

// Export cache blocking parameters for benchmarking/debugging
pub const MC = config.MC;
pub const KC = config.KC;
pub const NC = config.NC;
pub const MR = config.MR;
pub const NR = config.NR;

// ============================================================================
// Tests
// ============================================================================

test "sgemm basic" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemm(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 19.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 22.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 43.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 50.0, 1e-5);
}

test "sgemv basic" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 0, 0 };

    sgemv(2, 3, &A, &x, &y, 1.0, 0.0);

    try std.testing.expectApproxEqRel(y[0], 14.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 32.0, 1e-5);
}

test "getInfo" {
    const info = getInfo();
    try std.testing.expect(info.mr > 0);
    try std.testing.expect(info.nr > 0);
}
