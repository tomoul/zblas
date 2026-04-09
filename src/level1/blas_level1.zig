// zblas/src/level1/blas_level1.zig
// SIMD-optimized Level 1 BLAS operations
//
// All operations support unit stride (packed arrays).
// BLAS-style incx/incy stride parameters are not needed for our use cases
// (Tomoul tensors are always contiguous).

const std = @import("std");
const config = @import("../config.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);

// ============================================================================
// SAXPY: y = alpha * x + y
// ============================================================================

pub fn saxpy(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    if (n == 0 or alpha == 0.0) return;
    std.debug.assert(x.len >= n);
    std.debug.assert(y.len >= n);

    const alpha_vec: Vec = @splat(alpha);
    var i: usize = 0;

    // Unroll 4× for instruction-level parallelism
    while (i + 4 * VEC_WIDTH <= n) : (i += 4 * VEC_WIDTH) {
        inline for (0..4) |u| {
            const off = u * VEC_WIDTH;
            const x_vec: Vec = x[i + off ..][0..VEC_WIDTH].*;
            var y_vec: Vec = y[i + off ..][0..VEC_WIDTH].*;
            y_vec += alpha_vec * x_vec;
            y[i + off ..][0..VEC_WIDTH].* = y_vec;
        }
    }

    // Single-vector tail
    while (i + VEC_WIDTH <= n) : (i += VEC_WIDTH) {
        const x_vec: Vec = x[i..][0..VEC_WIDTH].*;
        var y_vec: Vec = y[i..][0..VEC_WIDTH].*;
        y_vec += alpha_vec * x_vec;
        y[i..][0..VEC_WIDTH].* = y_vec;
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        y[i] += alpha * x[i];
    }
}

// ============================================================================
// SDOT: dot = x · y
// ============================================================================

pub fn sdot(n: usize, x: []const f32, y: []const f32) f32 {
    if (n == 0) return 0.0;
    std.debug.assert(x.len >= n);
    std.debug.assert(y.len >= n);

    var i: usize = 0;

    // 4-way accumulator to reduce dependency chains
    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);

    while (i + 4 * VEC_WIDTH <= n) : (i += 4 * VEC_WIDTH) {
        inline for (0..4) |u| {
            const off = u * VEC_WIDTH;
            const x_vec: Vec = x[i + off ..][0..VEC_WIDTH].*;
            const y_vec: Vec = y[i + off ..][0..VEC_WIDTH].*;
            switch (u) {
                0 => acc0 += x_vec * y_vec,
                1 => acc1 += x_vec * y_vec,
                2 => acc2 += x_vec * y_vec,
                3 => acc3 += x_vec * y_vec,
                else => unreachable,
            }
        }
    }

    // Merge 4 accumulators
    acc0 = (acc0 + acc1) + (acc2 + acc3);

    // Single-vector tail
    while (i + VEC_WIDTH <= n) : (i += VEC_WIDTH) {
        const x_vec: Vec = x[i..][0..VEC_WIDTH].*;
        const y_vec: Vec = y[i..][0..VEC_WIDTH].*;
        acc0 += x_vec * y_vec;
    }

    var sum = @reduce(.Add, acc0);

    // Scalar tail
    while (i < n) : (i += 1) {
        sum += x[i] * y[i];
    }

    return sum;
}

// ============================================================================
// SSCAL: x = alpha * x
// ============================================================================

pub fn sscal(n: usize, alpha: f32, x: []f32) void {
    if (n == 0) return;
    std.debug.assert(x.len >= n);

    // Special cases
    if (alpha == 1.0) return;
    if (alpha == 0.0) {
        @memset(x[0..n], 0.0);
        return;
    }

    const alpha_vec: Vec = @splat(alpha);
    var i: usize = 0;

    while (i + 4 * VEC_WIDTH <= n) : (i += 4 * VEC_WIDTH) {
        inline for (0..4) |u| {
            const off = u * VEC_WIDTH;
            const v: Vec = x[i + off ..][0..VEC_WIDTH].*;
            x[i + off ..][0..VEC_WIDTH].* = v * alpha_vec;
        }
    }

    while (i + VEC_WIDTH <= n) : (i += VEC_WIDTH) {
        const v: Vec = x[i..][0..VEC_WIDTH].*;
        x[i..][0..VEC_WIDTH].* = v * alpha_vec;
    }

    while (i < n) : (i += 1) {
        x[i] *= alpha;
    }
}

// ============================================================================
// SNRM2: result = ||x||_2  (Euclidean norm)
// ============================================================================

pub fn snrm2(n: usize, x: []const f32) f32 {
    if (n == 0) return 0.0;
    std.debug.assert(x.len >= n);

    var i: usize = 0;
    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);

    while (i + 4 * VEC_WIDTH <= n) : (i += 4 * VEC_WIDTH) {
        inline for (0..4) |u| {
            const off = u * VEC_WIDTH;
            const v: Vec = x[i + off ..][0..VEC_WIDTH].*;
            switch (u) {
                0 => acc0 += v * v,
                1 => acc1 += v * v,
                2 => acc2 += v * v,
                3 => acc3 += v * v,
                else => unreachable,
            }
        }
    }

    acc0 = (acc0 + acc1) + (acc2 + acc3);

    while (i + VEC_WIDTH <= n) : (i += VEC_WIDTH) {
        const v: Vec = x[i..][0..VEC_WIDTH].*;
        acc0 += v * v;
    }

    var sum = @reduce(.Add, acc0);

    while (i < n) : (i += 1) {
        sum += x[i] * x[i];
    }

    return @sqrt(sum);
}

// ============================================================================
// SCOPY: y = x
// ============================================================================

pub fn scopy(n: usize, x: []const f32, y: []f32) void {
    if (n == 0) return;
    std.debug.assert(x.len >= n);
    std.debug.assert(y.len >= n);
    @memcpy(y[0..n], x[0..n]);
}

// ============================================================================
// ISAMAX: index of element with max absolute value
// ============================================================================

pub fn isamax(n: usize, x: []const f32) usize {
    if (n == 0) return 0;
    std.debug.assert(x.len >= n);

    var max_val = @abs(x[0]);
    var max_idx: usize = 0;

    // For short vectors, scalar is fine (no SIMD reduction for index tracking)
    // SIMD max-with-index is complex and the payoff is small since isamax
    // is rarely on the critical path in transformer inference.
    for (1..n) |i| {
        const val = @abs(x[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    return max_idx;
}

// ============================================================================
// Tests
// ============================================================================

test "saxpy basic" {
    var y = [_]f32{ 1, 2, 3, 4 };
    const x = [_]f32{ 10, 20, 30, 40 };
    saxpy(4, 2.0, &x, &y);
    try std.testing.expectApproxEqRel(y[0], 21.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 42.0, 1e-5);
    try std.testing.expectApproxEqRel(y[2], 63.0, 1e-5);
    try std.testing.expectApproxEqRel(y[3], 84.0, 1e-5);
}

test "saxpy alpha zero" {
    var y = [_]f32{ 1, 2, 3 };
    const x = [_]f32{ 10, 20, 30 };
    saxpy(3, 0.0, &x, &y);
    try std.testing.expectEqual(y[0], 1.0);
    try std.testing.expectEqual(y[1], 2.0);
}

test "saxpy large SIMD" {
    const N = 256;
    var x: [N]f32 = undefined;
    var y: [N]f32 = undefined;
    var y_ref: [N]f32 = undefined;
    for (0..N) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.1;
        y[i] = @as(f32, @floatFromInt(i)) * 0.2;
        y_ref[i] = y[i];
    }
    saxpy(N, 3.0, &x, &y);
    for (0..N) |i| y_ref[i] += 3.0 * x[i];
    for (0..N) |i| {
        try std.testing.expectApproxEqAbs(y[i], y_ref[i], 1e-4);
    }
}

test "sdot basic" {
    const x = [_]f32{ 1, 2, 3 };
    const y = [_]f32{ 4, 5, 6 };
    const result = sdot(3, &x, &y);
    try std.testing.expectApproxEqRel(result, 32.0, 1e-5);
}

test "sdot large SIMD" {
    const N = 384;
    var x: [N]f32 = undefined;
    var y: [N]f32 = undefined;
    for (0..N) |i| {
        x[i] = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.8;
        y[i] = @as(f32, @floatFromInt(i % 13)) * 0.1 - 0.6;
    }
    const result = sdot(N, &x, &y);
    var ref: f32 = 0.0;
    for (0..N) |i| ref += x[i] * y[i];
    try std.testing.expectApproxEqAbs(result, ref, 1e-3);
}

test "sscal basic" {
    var x = [_]f32{ 1, 2, 3, 4 };
    sscal(4, 3.0, &x);
    try std.testing.expectApproxEqRel(x[0], 3.0, 1e-5);
    try std.testing.expectApproxEqRel(x[3], 12.0, 1e-5);
}

test "sscal zero" {
    var x = [_]f32{ 1, 2, 3 };
    sscal(3, 0.0, &x);
    try std.testing.expectEqual(x[0], 0.0);
    try std.testing.expectEqual(x[2], 0.0);
}

test "sscal one noop" {
    var x = [_]f32{ 5, 10 };
    sscal(2, 1.0, &x);
    try std.testing.expectEqual(x[0], 5.0);
}

test "snrm2 basic" {
    const x = [_]f32{ 3, 4 };
    const result = snrm2(2, &x);
    try std.testing.expectApproxEqRel(result, 5.0, 1e-5);
}

test "snrm2 large SIMD" {
    const N = 384;
    var x: [N]f32 = undefined;
    for (0..N) |i| x[i] = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;
    const result = snrm2(N, &x);
    var ref: f32 = 0.0;
    for (0..N) |i| ref += x[i] * x[i];
    const ref_nrm = @sqrt(ref);
    try std.testing.expectApproxEqAbs(result, ref_nrm, 1e-3);
}

test "scopy basic" {
    const x = [_]f32{ 1, 2, 3, 4 };
    var y = [_]f32{ 0, 0, 0, 0 };
    scopy(4, &x, &y);
    try std.testing.expectEqual(y[0], 1.0);
    try std.testing.expectEqual(y[3], 4.0);
}

test "isamax basic" {
    const x = [_]f32{ 1, -5, 3, 2 };
    const idx = isamax(4, &x);
    try std.testing.expectEqual(idx, 1);
}

test "isamax all negative" {
    const x = [_]f32{ -1, -10, -3 };
    const idx = isamax(3, &x);
    try std.testing.expectEqual(idx, 1);
}

test "isamax single" {
    const x = [_]f32{42.0};
    try std.testing.expectEqual(isamax(1, &x), 0);
}

test "isamax empty" {
    try std.testing.expectEqual(isamax(0, &[_]f32{}), 0);
}
