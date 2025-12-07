// zblas/src/level3/sgemm.zig
// Single-precision General Matrix Multiply (SGEMM)
//
// This is the main SGEMM dispatcher that selects between:
// - Reference implementation (for small matrices or testing)
// - Cache-blocked implementation with optimized micro-kernels
//
// Phase 2: Cache-blocked implementation with 4x4 generic micro-kernel

const std = @import("std");
const config = @import("../config.zig");
const reference = @import("../reference.zig");
const packing = @import("../util/packing.zig");
const micro_kernel = @import("../kernel/generic/sgemm_kernel_4x4.zig");

// Re-export Transpose type from parent
pub const Transpose = @import("../zblas.zig").Transpose;

/// SGEMM: C = alpha * A * B + beta * C
/// A is [M x K], B is [K x N], C is [M x N], all row-major
pub fn sgemm(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    // Validate input sizes
    if (M == 0 or N == 0 or K == 0) return;
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B.len >= K * N);
    std.debug.assert(C.len >= M * N);

    // For now, use reference implementation
    // TODO Phase 2: Add cache-blocked implementation with threshold
    if (M * N * K < config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE) {
        reference.sgemm_reference_simple(M, N, K, A, B, C, alpha, beta);
        return;
    }

    // Use optimized path (currently just reference, will be replaced)
    sgemmOptimized(M, N, K, A, K, B, N, C, N, alpha, beta);
}

/// SGEMM with leading dimension parameters - cache-blocked implementation
fn sgemmOptimized(
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
    // Cache blocking parameters
    const MC = config.MC;
    const KC = config.KC;
    const NC = config.NC;
    const MR = micro_kernel.mr;
    const NR = micro_kernel.nr;

    // Allocate packing buffers on stack
    // For generic: MC=128, KC=256 -> 128*256 = 32KB for A
    // KC=256, NC=1024 -> 256*1024 = 256KB for B
    var packed_a: [MC * KC]f32 = undefined;
    var packed_b: [KC * NC]f32 = undefined;

    // Scale C by beta first (applied once before all accumulation)
    if (beta == 0.0) {
        for (0..M) |i| {
            @memset(C[i * ldc ..][0..N], 0.0);
        }
    } else if (beta != 1.0) {
        for (0..M) |i| {
            for (0..N) |j| {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Main blocking loops (GotoBLAS algorithm)
    // Loop order: jc (NC blocks), pc (KC blocks), ic (MC blocks)
    var jc: usize = 0;
    while (jc < N) : (jc += NC) {
        const jb = @min(NC, N - jc); // Actual columns in this block

        var pc: usize = 0;
        while (pc < K) : (pc += KC) {
            const pb = @min(KC, K - pc); // Actual K in this block

            // Pack B panel [pb × jb] -> packed_b
            // B block starts at B[pc * ldb + jc]
            packing.packB(
                B[pc * ldb + jc ..],
                ldb,
                &packed_b,
                pb,
                jb,
                NR,
            );

            var ic: usize = 0;
            while (ic < M) : (ic += MC) {
                const ib = @min(MC, M - ic); // Actual rows in this block

                // Pack A block [ib × pb] -> packed_a
                // A block starts at A[ic * lda + pc]
                packing.packA(
                    A[ic * lda + pc ..],
                    lda,
                    &packed_a,
                    ib,
                    pb,
                    MR,
                );

                // Compute C[ic:ic+ib, jc:jc+jb] += packed_A * packed_B
                computeBlock(
                    &packed_a,
                    &packed_b,
                    C[ic * ldc + jc ..],
                    ldc,
                    ib,
                    jb,
                    pb,
                    alpha,
                );
            }
        }
    }
}

/// Compute a block of C using packed A and B matrices
fn computeBlock(
    packed_a: []const f32,
    packed_b: []const f32,
    C: []f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) void {
    const MR = micro_kernel.mr;
    const NR = micro_kernel.nr;

    // Process NR×MR micro-tiles
    var jr: usize = 0;
    while (jr < n) : (jr += NR) {
        const nr_actual = @min(NR, n - jr);

        var ir: usize = 0;
        while (ir < m) : (ir += MR) {
            const mr_actual = @min(MR, m - ir);

            if (mr_actual == MR and nr_actual == NR) {
                // Full micro-kernel (4×4)
                // A panel for row ir is at packed_a[ir * k ..]
                // B panel for col jr is at packed_b[jr * k ..]
                micro_kernel.kernel(
                    k,
                    @ptrCast(packed_a.ptr + ir * k),
                    @ptrCast(packed_b.ptr + jr * k),
                    @ptrCast(C.ptr + ir * ldc + jr),
                    ldc,
                    alpha,
                );
            } else {
                // Edge case: use scalar for partial tiles
                scalarMicroKernel(
                    packed_a[ir * k ..],
                    packed_b[jr * k ..],
                    C[ir * ldc + jr ..],
                    ldc,
                    mr_actual,
                    nr_actual,
                    k,
                    alpha,
                    MR,
                    NR,
                );
            }
        }
    }
}

/// Scalar micro-kernel for edge cases (partial tiles)
fn scalarMicroKernel(
    A: []const f32,
    B: []const f32,
    C: []f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    mr: usize,
    nr: usize,
) void {
    // A is packed: for each kk, elements [kk*mr .. kk*mr + m] contain the m rows
    // B is packed: for each kk, elements [kk*nr .. kk*nr + n] contain the n cols
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                // A[i] at position kk is at A[kk * mr + i]
                // B[j] at position kk is at B[kk * nr + j]
                sum += A[kk * mr + i] * B[kk * nr + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with transpose options
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
    // Handle transpose cases
    // For now, implement NN case and error on others
    // TODO: Full transpose support in Phase 2+

    if (transA == .NoTrans and transB == .NoTrans) {
        // NN case: straightforward
        sgemmOptimized(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else if (transA == .Trans and transB == .NoTrans) {
        // TN case: A is transposed
        sgemmTN(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else if (transA == .NoTrans and transB == .Trans) {
        // NT case: B is transposed
        sgemmNT(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else {
        // TT case: both transposed
        sgemmTT(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    }
}

/// SGEMM with A transposed (TN case)
fn sgemmTN(
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
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A^T * B
    // A^T[i,k] = A[k,i], so A is K x M and we read A[k * lda + i]
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with B transposed (NT case)
fn sgemmNT(
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
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A * B^T
    // B^T[k,j] = B[j,k], so B is N x K and we read B[j * ldb + k]
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with both transposed (TT case)
fn sgemmTT(
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
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A^T * B^T
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[k * lda + i] * B[j * ldb + k];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sgemm NN" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemm(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 19.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 22.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 43.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 50.0, 1e-5);
}

test "sgemm TN" {
    // A^T = [1, 3; 2, 4] (original A = [1, 2; 3, 4])
    // B = [5, 6; 7, 8]
    // A^T * B = [26, 30; 38, 44]
    const A = [_]f32{ 1, 2, 3, 4 }; // stored as 2x2, will be transposed
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemmTranspose(.Trans, .NoTrans, 2, 2, 2, &A, 2, &B, 2, &C, 2, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 26.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 30.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 38.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 44.0, 1e-5);
}

test "sgemm NT" {
    // A = [1, 2; 3, 4]
    // B^T = [5, 7; 6, 8] (original B = [5, 6; 7, 8])
    // A * B^T = [17, 23; 39, 53]
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemmTranspose(.NoTrans, .Trans, 2, 2, 2, &A, 2, &B, 2, &C, 2, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 17.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 23.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 39.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 53.0, 1e-5);
}

test "sgemm empty" {
    var C = [_]f32{ 1, 2, 3, 4 };
    sgemm(0, 2, 2, &[_]f32{}, &[_]f32{}, &C, 1.0, 0.0);
    // Should not modify C
    try std.testing.expectEqual(C[0], 1.0);
}
