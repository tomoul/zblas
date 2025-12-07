// zblas/src/level2/sgemv.zig
// Single-precision General Matrix-Vector Multiply (SGEMV)
//
// y = alpha * A * x + beta * y
// A is [M x N], x is [N], y is [M]

const std = @import("std");
const config = @import("../config.zig");
const reference = @import("../reference.zig");

/// SGEMV: y = alpha * A * x + beta * y
/// A is [M x N] row-major, x is [N], y is [M]
pub fn sgemv(
    M: usize,
    N: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    // Validate input sizes
    if (M == 0 or N == 0) return;
    std.debug.assert(A.len >= M * N);
    std.debug.assert(x.len >= N);
    std.debug.assert(y.len >= M);

    // For now, use reference implementation
    // TODO Phase 5: Add SIMD-optimized version
    if (N < config.MIN_OPTIMIZED_SIZE) {
        reference.sgemv_reference_simple(M, N, A, x, y, alpha, beta);
        return;
    }

    // Use SIMD-optimized path
    sgemvOptimized(M, N, A, N, x, y, alpha, beta);
}

/// SGEMV with leading dimension parameter
fn sgemvOptimized(
    M: usize,
    N: usize,
    A: []const f32,
    lda: usize,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    const VEC_WIDTH = comptime config.getVectorWidth();
    const Vec = @Vector(VEC_WIDTH, f32);

    // Scale y by beta
    if (beta == 0.0) {
        @memset(y[0..M], 0.0);
    } else if (beta != 1.0) {
        for (y[0..M]) |*val| {
            val.* *= beta;
        }
    }

    // If alpha is zero, we're done
    if (alpha == 0.0) return;

    // Process each row
    for (0..M) |i| {
        var sum: f32 = 0.0;
        var j: usize = 0;

        // SIMD loop
        if (VEC_WIDTH > 1 and N >= VEC_WIDTH) {
            var sum_vec: Vec = @splat(0.0);

            while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
                const a_vec: Vec = A[i * lda + j ..][0..VEC_WIDTH].*;
                const x_vec: Vec = x[j..][0..VEC_WIDTH].*;
                sum_vec += a_vec * x_vec;
            }

            // Reduce vector to scalar
            sum = @reduce(.Add, sum_vec);
        }

        // Scalar tail
        while (j < N) : (j += 1) {
            sum += A[i * lda + j] * x[j];
        }

        y[i] += alpha * sum;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sgemv basic" {
    // A = [1, 2, 3; 4, 5, 6], x = [1, 2, 3]
    // y = A * x = [14, 32]
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 0, 0 };

    sgemv(2, 3, &A, &x, &y, 1.0, 0.0);

    try std.testing.expectApproxEqRel(y[0], 14.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 32.0, 1e-5);
}

test "sgemv with alpha beta" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 1, 1 };

    // y = 2.0 * A * x + 0.5 * y = 2*[14, 32] + 0.5*[1, 1] = [28.5, 64.5]
    sgemv(2, 3, &A, &x, &y, 2.0, 0.5);

    try std.testing.expectApproxEqRel(y[0], 28.5, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 64.5, 1e-5);
}

test "sgemv empty" {
    var y = [_]f32{ 1, 2 };
    sgemv(0, 3, &[_]f32{}, &[_]f32{}, &y, 1.0, 0.0);
    // Should not modify y
    try std.testing.expectEqual(y[0], 1.0);
}

test "sgemv large" {
    // Test with size > MIN_OPTIMIZED_SIZE to exercise SIMD path
    const N = 64;
    const M = 4;
    var A: [M * N]f32 = undefined;
    var x: [N]f32 = undefined;
    var y: [M]f32 = undefined;
    var y_ref: [M]f32 = undefined;

    // Initialize with simple pattern
    for (0..M) |i| {
        for (0..N) |j| {
            A[i * N + j] = @as(f32, @floatFromInt(i + j));
        }
    }
    for (0..N) |j| {
        x[j] = @as(f32, @floatFromInt(j));
    }
    @memset(&y, 0.0);
    @memset(&y_ref, 0.0);

    // Compute with optimized path
    sgemv(M, N, &A, &x, &y, 1.0, 0.0);

    // Compute with reference
    reference.sgemv_reference_simple(M, N, &A, &x, &y_ref, 1.0, 0.0);

    // Compare
    for (0..M) |i| {
        try std.testing.expectApproxEqRel(y[i], y_ref[i], 1e-5);
    }
}
