// zblas/tests/test_sgemm.zig
// Comprehensive correctness tests for SGEMM

const std = @import("std");
const zblas = @import("zblas");

// ============================================================================
// Basic Correctness Tests
// ============================================================================

test "sgemm 2x2 identity" {
    // C = I * I = I
    const I = [_]f32{ 1, 0, 0, 1 };
    var C = [_]f32{ 0, 0, 0, 0 };

    zblas.sgemm(2, 2, 2, &I, &I, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 1.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 0.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 0.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 1.0, 1e-5);
}

test "sgemm 2x2 known result" {
    // A = [1, 2; 3, 4], B = [5, 6; 7, 8]
    // C = A * B = [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    zblas.sgemm(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 19.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 22.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 43.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 50.0, 1e-5);
}

test "sgemm 3x3 known result" {
    // A = [1, 2, 3; 4, 5, 6; 7, 8, 9]
    // B = [9, 8, 7; 6, 5, 4; 3, 2, 1]
    // C = A * B = [30, 24, 18; 84, 69, 54; 138, 114, 90]
    const A = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const B = [_]f32{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    var C = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    zblas.sgemm(3, 3, 3, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 30.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 24.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 18.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 84.0, 1e-5);
    try std.testing.expectApproxEqRel(C[4], 69.0, 1e-5);
    try std.testing.expectApproxEqRel(C[5], 54.0, 1e-5);
    try std.testing.expectApproxEqRel(C[6], 138.0, 1e-5);
    try std.testing.expectApproxEqRel(C[7], 114.0, 1e-5);
    try std.testing.expectApproxEqRel(C[8], 90.0, 1e-5);
}

// ============================================================================
// Non-square Matrix Tests
// ============================================================================

test "sgemm 2x3 * 3x2" {
    // A = [1, 2, 3; 4, 5, 6] (2x3)
    // B = [7, 8; 9, 10; 11, 12] (3x2)
    // C = A * B = [58, 64; 139, 154] (2x2)
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var C = [_]f32{ 0, 0, 0, 0 };

    zblas.sgemm(2, 2, 3, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 58.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 64.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 139.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 154.0, 1e-5);
}

test "sgemm 1xN * Nx1 (vector outer product)" {
    // A = [1, 2, 3] (1x3)
    // B = [4; 5; 6] (3x1)
    // C = A * B = [32] (1x1)
    const A = [_]f32{ 1, 2, 3 };
    const B = [_]f32{ 4, 5, 6 };
    var C = [_]f32{0};

    zblas.sgemm(1, 1, 3, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 32.0, 1e-5);
}

// ============================================================================
// Alpha and Beta Scaling Tests
// ============================================================================

test "sgemm alpha scaling" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    // C = 2.0 * A * B = 2 * [19, 22; 43, 50] = [38, 44; 86, 100]
    zblas.sgemm(2, 2, 2, &A, &B, &C, 2.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 38.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 44.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 86.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 100.0, 1e-5);
}

test "sgemm beta accumulation" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 10, 10, 10, 10 };

    // C = 1.0 * A * B + 1.0 * C = [19, 22; 43, 50] + [10, 10; 10, 10]
    zblas.sgemm(2, 2, 2, &A, &B, &C, 1.0, 1.0);

    try std.testing.expectApproxEqRel(C[0], 29.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 32.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 53.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 60.0, 1e-5);
}

test "sgemm alpha and beta combined" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 1, 1, 1, 1 };

    // C = 2.0 * A * B + 0.5 * C
    zblas.sgemm(2, 2, 2, &A, &B, &C, 2.0, 0.5);

    // Expected: 2*[19,22;43,50] + 0.5*[1,1;1,1] = [38.5, 44.5; 86.5, 100.5]
    try std.testing.expectApproxEqRel(C[0], 38.5, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 44.5, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 86.5, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 100.5, 1e-5);
}

test "sgemm beta zero overwrites" {
    const A = [_]f32{ 1, 0, 0, 1 };
    const B = [_]f32{ 1, 0, 0, 1 };
    var C = [_]f32{ 999, 999, 999, 999 };

    zblas.sgemm(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    // C should be identity, not affected by initial values
    try std.testing.expectApproxEqRel(C[0], 1.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 0.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 0.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 1.0, 1e-5);
}

// ============================================================================
// Edge Cases
// ============================================================================

test "sgemm M=1" {
    const A = [_]f32{ 1, 2, 3 }; // 1x3
    const B = [_]f32{ 4, 5, 6, 7, 8, 9 }; // 3x2
    var C = [_]f32{ 0, 0 }; // 1x2

    zblas.sgemm(1, 2, 3, &A, &B, &C, 1.0, 0.0);

    // C = [1*4+2*6+3*8, 1*5+2*7+3*9] = [40, 46]
    try std.testing.expectApproxEqRel(C[0], 40.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 46.0, 1e-5);
}

test "sgemm N=1" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2x3
    const B = [_]f32{ 7, 8, 9 }; // 3x1
    var C = [_]f32{ 0, 0 }; // 2x1

    zblas.sgemm(2, 1, 3, &A, &B, &C, 1.0, 0.0);

    // C = [1*7+2*8+3*9; 4*7+5*8+6*9] = [50; 122]
    try std.testing.expectApproxEqRel(C[0], 50.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 122.0, 1e-5);
}

test "sgemm K=1" {
    const A = [_]f32{ 1, 2 }; // 2x1
    const B = [_]f32{ 3, 4 }; // 1x2
    var C = [_]f32{ 0, 0, 0, 0 }; // 2x2

    zblas.sgemm(2, 2, 1, &A, &B, &C, 1.0, 0.0);

    // C = [1*3, 1*4; 2*3, 2*4] = [3, 4; 6, 8]
    try std.testing.expectApproxEqRel(C[0], 3.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 4.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 6.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 8.0, 1e-5);
}

// ============================================================================
// Larger Matrix Tests (compared against reference)
// ============================================================================

test "sgemm 16x16 against reference" {
    const N = 16;
    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    // Initialize with simple pattern
    for (0..N) |i| {
        for (0..N) |j| {
            A[i * N + j] = @as(f32, @floatFromInt((i + j) % 10));
            B[i * N + j] = @as(f32, @floatFromInt((i * j) % 10));
        }
    }
    @memset(&C, 0.0);
    @memset(&C_ref, 0.0);

    // Compute with zblas
    zblas.sgemm(N, N, N, &A, &B, &C, 1.0, 0.0);

    // Compute with reference
    zblas.reference.sgemm_reference_simple(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    // Compare
    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C[i], C_ref[i], 1e-4);
    }
}

test "sgemm 64x64 against reference" {
    const N = 64;
    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    // Initialize with random-ish pattern
    for (0..N) |i| {
        for (0..N) |j| {
            A[i * N + j] = @as(f32, @floatFromInt((i * 7 + j * 11) % 100)) / 100.0;
            B[i * N + j] = @as(f32, @floatFromInt((i * 13 + j * 17) % 100)) / 100.0;
        }
    }
    @memset(&C, 0.0);
    @memset(&C_ref, 0.0);

    zblas.sgemm(N, N, N, &A, &B, &C, 1.0, 0.0);
    zblas.reference.sgemm_reference_simple(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C[i], C_ref[i], 1e-3);
    }
}

// ============================================================================
// SGEMV Tests
// ============================================================================

test "sgemv basic" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 0, 0 };

    zblas.sgemv(2, 3, &A, &x, &y, 1.0, 0.0);

    try std.testing.expectApproxEqRel(y[0], 14.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 32.0, 1e-5);
}

test "sgemv with alpha beta" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 1, 1 };

    zblas.sgemv(2, 3, &A, &x, &y, 2.0, 0.5);

    try std.testing.expectApproxEqRel(y[0], 28.5, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 64.5, 1e-5);
}

// ============================================================================
// Parallel SGEMM Tests (Phase 6)
// ============================================================================

test "parallel sgemm 64x64 correctness" {
    const allocator = std.testing.allocator;
    const N = 64;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_parallel: [N * N]f32 = undefined;
    var C_single: [N * N]f32 = undefined;

    // Initialize with deterministic values
    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_parallel, 0.0);
    @memset(&C_single, 0.0);

    // Single-threaded reference
    zblas.sgemm(N, N, N, &A, &B, &C_single, 1.0, 0.0);

    // Parallel with 4 threads (will use single-threaded due to size threshold)
    try zblas.sgemmParallelN(allocator, 4, N, N, N, &A, &B, &C_parallel, 1.0, 0.0);

    // Compare results
    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm 128x128 correctness" {
    const allocator = std.testing.allocator;
    const N = 128;

    const A = try allocator.alloc(f32, N * N);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, N * N);
    defer allocator.free(B);
    const C_parallel = try allocator.alloc(f32, N * N);
    defer allocator.free(C_parallel);
    const C_single = try allocator.alloc(f32, N * N);
    defer allocator.free(C_single);

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(C_parallel, 0.0);
    @memset(C_single, 0.0);

    zblas.sgemm(N, N, N, A, B, C_single, 1.0, 0.0);
    try zblas.sgemmParallelN(allocator, 4, N, N, N, A, B, C_parallel, 1.0, 0.0);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm with beta" {
    const allocator = std.testing.allocator;
    const N = 64;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_parallel: [N * N]f32 = undefined;
    var C_single: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 50)) / 50.0;
        B[i] = @as(f32, @floatFromInt((i * 3) % 50)) / 50.0;
        C_parallel[i] = @as(f32, @floatFromInt((i * 11) % 50)) / 50.0;
        C_single[i] = C_parallel[i];
    }

    // Test with alpha=2.0, beta=0.5
    zblas.sgemm(N, N, N, &A, &B, &C_single, 2.0, 0.5);
    try zblas.sgemmParallelN(allocator, 4, N, N, N, &A, &B, &C_parallel, 2.0, 0.5);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm various thread counts" {
    const allocator = std.testing.allocator;
    const N = 64;
    const thread_counts = [_]usize{ 1, 2, 3, 4, 7, 8 };

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_ref, 0.0);

    // Get reference result
    zblas.sgemm(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    // Test each thread count
    for (thread_counts) |num_threads| {
        var C_test: [N * N]f32 = undefined;
        @memset(&C_test, 0.0);

        try zblas.sgemmParallelN(allocator, num_threads, N, N, N, &A, &B, &C_test, 1.0, 0.0);

        for (0..N * N) |i| {
            try std.testing.expectApproxEqRel(C_test[i], C_ref[i], 1e-4);
        }
    }
}

test "shouldParallelize thresholds" {
    // Small matrix - should NOT parallelize
    try std.testing.expect(!zblas.shouldParallelize(100, 100, 100));

    // Large matrix - should parallelize
    try std.testing.expect(zblas.shouldParallelize(1024, 1024, 1024));
}

test "getOptimalThreadCount" {
    // Small matrix - 1 thread
    try std.testing.expectEqual(@as(usize, 1), zblas.getOptimalThreadCount(100, 100, 100));

    // Large matrix - multiple threads
    const threads = zblas.getOptimalThreadCount(2048, 2048, 2048);
    try std.testing.expect(threads >= 1);
}

test "context sgemm" {
    const allocator = std.testing.allocator;
    const N = 64;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_ctx: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_ctx, 0.0);
    @memset(&C_ref, 0.0);

    // Reference
    zblas.sgemm(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    // Context-based (single threaded)
    var ctx = zblas.Context.init(allocator, 1);
    try ctx.sgemm(N, N, N, &A, &B, &C_ctx, 1.0, 0.0);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_ctx[i], C_ref[i], 1e-4);
    }
}

test "context sgemm multi-threaded" {
    const allocator = std.testing.allocator;
    const N = 64;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_ctx: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_ctx, 0.0);
    @memset(&C_ref, 0.0);

    // Reference
    zblas.sgemm(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    // Context-based with 4 threads (will fall back to single for small N)
    var ctx = zblas.Context.init(allocator, 4);
    try ctx.sgemm(N, N, N, &A, &B, &C_ctx, 1.0, 0.0);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_ctx[i], C_ref[i], 1e-4);
    }
}
