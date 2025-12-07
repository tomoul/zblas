const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ==========================================================================
    // zblas Library Module
    // ==========================================================================
    const zblas_module = b.addModule("zblas", .{
        .root_source_file = b.path("src/zblas.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ==========================================================================
    // Static Library
    // ==========================================================================
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/zblas.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zblas",
        .root_module = lib_module,
    });

    b.installArtifact(lib);

    // ==========================================================================
    // Unit Tests
    // ==========================================================================
    const test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_sgemm.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("zblas", zblas_module);

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run zblas unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ==========================================================================
    // Benchmarks
    // ==========================================================================
    const bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Always optimize benchmarks
    });
    bench_module.addImport("zblas", zblas_module);

    const benchmark = b.addExecutable(.{
        .name = "zblas-benchmark",
        .root_module = bench_module,
    });

    b.installArtifact(benchmark);

    const run_benchmark = b.addRunArtifact(benchmark);
    const bench_step = b.step("bench", "Run zblas benchmarks");
    bench_step.dependOn(&run_benchmark.step);

    // ==========================================================================
    // Reference Tests (verify against naive implementation)
    // ==========================================================================
    const ref_test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_reference.zig"),
        .target = target,
        .optimize = optimize,
    });
    ref_test_module.addImport("zblas", zblas_module);

    const ref_tests = b.addTest(.{
        .root_module = ref_test_module,
    });

    const run_ref_tests = b.addRunArtifact(ref_tests);
    const ref_test_step = b.step("test-ref", "Run reference correctness tests");
    ref_test_step.dependOn(&run_ref_tests.step);
}
