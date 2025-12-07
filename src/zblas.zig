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

// Import optimized implementations
// TODO: These will be replaced with optimized versions in later phases
const sgemm_impl = @import("level3/sgemm.zig");
const sgemv_impl = @import("level2/sgemv.zig");

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

// ============================================================================
// Level 1 BLAS (Vector Operations)
// ============================================================================

/// SAXPY: y = alpha * x + y
pub fn saxpy(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    // TODO: SIMD optimized version
    reference.saxpy_reference(n, alpha, x, y);
}

/// SDOT: dot product of two vectors
pub fn sdot(n: usize, x: []const f32, y: []const f32) f32 {
    // TODO: SIMD optimized version
    return reference.sdot_reference(n, x, y);
}

/// SSCAL: x = alpha * x
pub fn sscal(n: usize, alpha: f32, x: []f32) void {
    // TODO: SIMD optimized version
    reference.sscal_reference(n, alpha, x);
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
