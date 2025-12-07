// zblas/tests/test_reference.zig
// Tests for reference implementations

const std = @import("std");
const zblas = @import("zblas");
const reference = zblas.reference;

test "reference module accessible" {
    // Just verify we can access the reference module
    const info = zblas.getInfo();
    try std.testing.expect(info.mr > 0);
}

test "reference sgemm matches zblas" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var C1 = [_]f32{ 0, 0, 0, 0 };
    var C2 = [_]f32{ 0, 0, 0, 0 };

    reference.sgemm_reference_simple(2, 2, 3, &A, &B, &C1, 1.0, 0.0);
    zblas.sgemm(2, 2, 3, &A, &B, &C2, 1.0, 0.0);

    for (0..4) |i| {
        try std.testing.expectApproxEqRel(C1[i], C2[i], 1e-5);
    }
}

test "reference sdot" {
    const x = [_]f32{ 1, 2, 3, 4, 5 };
    const y = [_]f32{ 5, 4, 3, 2, 1 };

    const result = reference.sdot_reference(5, &x, &y);

    // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
    try std.testing.expectApproxEqRel(result, 35.0, 1e-5);
}

test "reference saxpy" {
    var y = [_]f32{ 1, 2, 3, 4, 5 };
    const x = [_]f32{ 10, 20, 30, 40, 50 };

    reference.saxpy_reference(5, 0.1, &x, &y);

    // y = 0.1 * x + y = [2, 4, 6, 8, 10]
    try std.testing.expectApproxEqRel(y[0], 2.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 4.0, 1e-5);
    try std.testing.expectApproxEqRel(y[2], 6.0, 1e-5);
    try std.testing.expectApproxEqRel(y[3], 8.0, 1e-5);
    try std.testing.expectApproxEqRel(y[4], 10.0, 1e-5);
}

test "reference sscal" {
    var x = [_]f32{ 1, 2, 3, 4, 5 };

    reference.sscal_reference(5, 2.0, &x);

    try std.testing.expectApproxEqRel(x[0], 2.0, 1e-5);
    try std.testing.expectApproxEqRel(x[1], 4.0, 1e-5);
    try std.testing.expectApproxEqRel(x[2], 6.0, 1e-5);
    try std.testing.expectApproxEqRel(x[3], 8.0, 1e-5);
    try std.testing.expectApproxEqRel(x[4], 10.0, 1e-5);
}
