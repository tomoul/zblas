// zblas/src/conv/conv1d.zig
//
// SIMD-optimized 1D convolution kernel
//
// Vectorizes over output channels (VEC_WIDTH at a time) for maximum SIMD
// utilization. Supports optional fused bias and ReLU to eliminate extra
// memory passes.
//
// Weight layout: The caller must repack weights from [C_out, C_in, K] to
// [C_in, K, C_out] using repackWeight() so that consecutive output channels
// are contiguous in memory for aligned SIMD loads.
//
// Shapes:
//   input:  [in_channels, in_width]       (row-major)
//   weight: [in_channels, kernel_size, out_channels]  (repacked for SIMD)
//   bias:   [out_channels]                (optional)
//   output: [out_channels, out_width]     (row-major)

const std = @import("std");
const config = @import("../config.zig");

const VEC_WIDTH = config.getVectorWidth();
const Vec = @Vector(VEC_WIDTH, f32);

/// Activation function to fuse into the convolution epilogue.
pub const Activation = enum {
    none,
    relu,
};

/// Repack convolution weights from standard [C_out, C_in, K] layout to
/// SIMD-friendly [C_in, K, C_out] layout.
///
/// Standard layout:  weight[oc][ic][k] = src[oc * (C_in * K) + ic * K + k]
/// Repacked layout:  weight[ic][k][oc] = dst[(ic * K + k) * C_out + oc]
///
/// This makes consecutive output channels contiguous, enabling aligned
/// VEC_WIDTH loads in the inner loop.
pub fn repackWeight(
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
    src: [*]const f32,
    dst: [*]f32,
) void {
    for (0..in_channels) |ic| {
        for (0..kernel_size) |k| {
            const dst_base = (ic * kernel_size + k) * out_channels;
            for (0..out_channels) |oc| {
                dst[dst_base + oc] = src[oc * (in_channels * kernel_size) + ic * kernel_size + k];
            }
        }
    }
}

/// SIMD Conv1d kernel.
///
/// Computes: output[oc, ow] = sum_{ic,k} input[ic, ow*stride+k-padding] * weight[ic,k,oc]
///           + bias[oc]  (if bias != null)
///           with optional fused activation
///
/// Weight must be in repacked [C_in, K, C_out] layout (use repackWeight).
pub fn conv1d(
    out_channels: usize,
    in_channels: usize,
    out_width: usize,
    in_width: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input: [*]const f32,
    weight: [*]const f32, // repacked: [C_in, K, C_out]
    bias: ?[*]const f32,
    output: [*]f32,
    comptime activation: Activation,
) void {
    const zero_vec: Vec = @splat(0.0);

    // Process each output spatial position
    for (0..out_width) |ow| {
        // SIMD loop: process VEC_WIDTH output channels at a time
        var oc: usize = 0;
        while (oc + VEC_WIDTH <= out_channels) : (oc += VEC_WIDTH) {
            var acc: Vec = zero_vec;

            // Accumulate across all input channels and kernel positions
            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    const in_pos_signed: i64 = @as(i64, @intCast(ow * stride + k)) - @as(i64, @intCast(padding));

                    if (in_pos_signed >= 0 and in_pos_signed < @as(i64, @intCast(in_width))) {
                        const in_pos: usize = @intCast(in_pos_signed);
                        // Broadcast input value
                        const in_val: Vec = @splat(input[ic * in_width + in_pos]);
                        // Load VEC_WIDTH weights (contiguous in repacked layout)
                        const w_offset = (ic * kernel_size + k) * out_channels + oc;
                        const w_vec: Vec = weight[w_offset..][0..VEC_WIDTH].*;
                        acc += in_val * w_vec;
                    }
                }
            }

            // Add bias if provided
            if (bias) |b| {
                const bias_vec: Vec = b[oc..][0..VEC_WIDTH].*;
                acc += bias_vec;
            }

            // Apply activation
            if (activation == .relu) {
                acc = @max(acc, zero_vec);
            }

            // Store: output is [out_channels, out_width], so output[oc..oc+VEC, ow]
            // is strided with stride=out_width. Must scatter-store.
            for (0..VEC_WIDTH) |v| {
                output[(oc + v) * out_width + ow] = acc[v];
            }
        }

        // Scalar tail for remaining output channels
        while (oc < out_channels) : (oc += 1) {
            var sum_val: f32 = 0.0;

            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    const in_pos_signed: i64 = @as(i64, @intCast(ow * stride + k)) - @as(i64, @intCast(padding));

                    if (in_pos_signed >= 0 and in_pos_signed < @as(i64, @intCast(in_width))) {
                        const in_pos: usize = @intCast(in_pos_signed);
                        const in_val = input[ic * in_width + in_pos];
                        const w_val = weight[(ic * kernel_size + k) * out_channels + oc];
                        sum_val += in_val * w_val;
                    }
                }
            }

            if (bias) |b| {
                sum_val += b[oc];
            }

            if (activation == .relu) {
                sum_val = @max(sum_val, 0.0);
            }

            output[oc * out_width + ow] = sum_val;
        }
    }
}

/// Optimized Conv1d for the no-padding case (avoids bounds checks in inner loop).
/// This covers STFT (padding=0) and any convolution where all input positions are valid.
pub fn conv1dNoPad(
    out_channels: usize,
    in_channels: usize,
    out_width: usize,
    in_width: usize,
    kernel_size: usize,
    stride: usize,
    input: [*]const f32,
    weight: [*]const f32, // repacked: [C_in, K, C_out]
    bias: ?[*]const f32,
    output: [*]f32,
    comptime activation: Activation,
) void {
    const zero_vec: Vec = @splat(0.0);

    for (0..out_width) |ow| {
        const base_pos = ow * stride;

        var oc: usize = 0;
        while (oc + VEC_WIDTH <= out_channels) : (oc += VEC_WIDTH) {
            var acc: Vec = zero_vec;

            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    const in_val: Vec = @splat(input[ic * in_width + base_pos + k]);
                    const w_offset = (ic * kernel_size + k) * out_channels + oc;
                    const w_vec: Vec = weight[w_offset..][0..VEC_WIDTH].*;
                    acc += in_val * w_vec;
                }
            }

            if (bias) |b| {
                const bias_vec: Vec = b[oc..][0..VEC_WIDTH].*;
                acc += bias_vec;
            }

            if (activation == .relu) {
                acc = @max(acc, zero_vec);
            }

            for (0..VEC_WIDTH) |v| {
                output[(oc + v) * out_width + ow] = acc[v];
            }
        }

        // Scalar tail
        while (oc < out_channels) : (oc += 1) {
            var sum_val: f32 = 0.0;
            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    sum_val += input[ic * in_width + base_pos + k] * weight[(ic * kernel_size + k) * out_channels + oc];
                }
            }
            if (bias) |b| sum_val += b[oc];
            if (activation == .relu) sum_val = @max(sum_val, 0.0);
            output[oc * out_width + ow] = sum_val;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "repackWeight basic" {
    // weight [2, 3, 2] (out=2, in=3, k=2)
    // Standard: weight[oc][ic][k]
    const src = [_]f32{
        // oc=0: ic=0,k=0; ic=0,k=1; ic=1,k=0; ic=1,k=1; ic=2,k=0; ic=2,k=1
        1, 2, 3, 4,  5,  6,
        // oc=1
        7, 8, 9, 10, 11, 12,
    };
    var dst: [12]f32 = undefined;
    repackWeight(2, 3, 2, &src, &dst);

    // Repacked: dst[(ic*K + k) * C_out + oc]
    // ic=0,k=0: oc=0 -> src[0*6+0*2+0]=1, oc=1 -> src[1*6+0*2+0]=7
    try testing.expectEqual(dst[0], 1.0); // (0,0,0)
    try testing.expectEqual(dst[1], 7.0); // (0,0,1)
    // ic=0,k=1: oc=0 -> src[0*6+0*2+1]=2, oc=1 -> src[1*6+0*2+1]=8
    try testing.expectEqual(dst[2], 2.0);
    try testing.expectEqual(dst[3], 8.0);
    // ic=1,k=0: oc=0 -> src[0*6+1*2+0]=3, oc=1 -> src[1*6+1*2+0]=9
    try testing.expectEqual(dst[4], 3.0);
    try testing.expectEqual(dst[5], 9.0);
}

test "conv1d basic no padding" {
    // Input [1, 4], Weight [2, 1, 3] -> Output [2, 2] with stride=1, padding=0
    const in_channels = 1;
    const out_channels = 2;
    const in_width = 4;
    const kernel_size = 3;
    const stride_val = 1;
    const out_width = (in_width - kernel_size) / stride_val + 1; // 2

    const input_data = [_]f32{ 1, 2, 3, 4 };

    // Standard weight [C_out=2, C_in=1, K=3]
    const weight_std = [_]f32{
        1, 0, -1, // oc=0
        0, 1, 0, // oc=1
    };

    // Repack to [C_in=1, K=3, C_out=2]
    var weight_repacked: [6]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    var output_data: [4]f32 = [_]f32{ 0, 0, 0, 0 };

    conv1d(
        out_channels,
        in_channels,
        out_width,
        in_width,
        kernel_size,
        stride_val,
        0, // no padding
        &input_data,
        &weight_repacked,
        null,
        &output_data,
        .none,
    );

    // oc=0: [1,0,-1] * [1,2,3] = 1-3 = -2; [1,0,-1] * [2,3,4] = 2-4 = -2
    // oc=1: [0,1,0] * [1,2,3] = 2; [0,1,0] * [2,3,4] = 3
    try testing.expectApproxEqAbs(output_data[0 * out_width + 0], -2.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[0 * out_width + 1], -2.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1 * out_width + 0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1 * out_width + 1], 3.0, 1e-5);
}

test "conv1d with padding" {
    // Input [1, 3], Weight [2, 1, 3], stride=1, padding=1 -> Output [2, 3]
    const in_channels = 1;
    const out_channels = 2;
    const in_width = 3;
    const kernel_size = 3;
    const stride_val = 1;
    const padding_val = 1;
    const out_width = (in_width + 2 * padding_val - kernel_size) / stride_val + 1; // 3

    const input_data = [_]f32{ 1, 2, 3 };
    const weight_std = [_]f32{
        1, 1, 1, // oc=0: sum filter
        0, 1, 0, // oc=1: identity
    };
    var weight_repacked: [6]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    var output_data: [6]f32 = undefined;

    conv1d(
        out_channels,
        in_channels,
        out_width,
        in_width,
        kernel_size,
        stride_val,
        padding_val,
        &input_data,
        &weight_repacked,
        null,
        &output_data,
        .none,
    );

    // oc=0 (sum): [pad,1,2]=3, [1,2,3]=6, [2,3,pad]=5
    try testing.expectApproxEqAbs(output_data[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1], 6.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[2], 5.0, 1e-5);

    // oc=1 (identity): [pad,1,2]->1, [1,2,3]->2, [2,3,pad]->3
    try testing.expectApproxEqAbs(output_data[3], 1.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[4], 2.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[5], 3.0, 1e-5);
}

test "conv1d with bias" {
    const in_channels = 1;
    const out_channels = 2;
    const in_width = 3;
    const kernel_size = 1;
    const out_width = 3;

    const input_data = [_]f32{ 1, 2, 3 };
    const weight_std = [_]f32{ 2, 3 }; // [2, 1, 1]
    var weight_repacked: [2]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    const bias_data = [_]f32{ 10, 20 };
    var output_data: [6]f32 = undefined;

    conv1d(out_channels, in_channels, out_width, in_width, kernel_size, 1, 0, &input_data, &weight_repacked, &bias_data, &output_data, .none);

    // oc=0: 2*x + 10 -> 12, 14, 16
    // oc=1: 3*x + 20 -> 23, 26, 29
    try testing.expectApproxEqAbs(output_data[0], 12.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1], 14.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[2], 16.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[3], 23.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[4], 26.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[5], 29.0, 1e-5);
}

test "conv1d with relu" {
    const in_channels = 1;
    const out_channels = 2;
    const in_width = 3;
    const kernel_size = 1;
    const out_width = 3;

    const input_data = [_]f32{ -1, 0, 1 };
    const weight_std = [_]f32{ 1, -1 }; // [2, 1, 1]
    var weight_repacked: [2]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    var output_data: [6]f32 = undefined;

    conv1d(out_channels, in_channels, out_width, in_width, kernel_size, 1, 0, &input_data, &weight_repacked, null, &output_data, .relu);

    // oc=0 (w=1): relu(-1)=0, relu(0)=0, relu(1)=1
    try testing.expectApproxEqAbs(output_data[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[2], 1.0, 1e-5);

    // oc=1 (w=-1): relu(1)=1, relu(0)=0, relu(-1)=0
    try testing.expectApproxEqAbs(output_data[3], 1.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[4], 0.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[5], 0.0, 1e-5);
}

test "conv1d stride 2 with padding" {
    // Input [1, 4], Weight [2, 1, 3], stride=2, padding=1 -> Output [2, 2]
    const in_channels = 1;
    const out_channels = 2;
    const in_width = 4;
    const kernel_size = 3;
    const stride_val = 2;
    const padding_val = 1;
    const out_width = (in_width + 2 * padding_val - kernel_size) / stride_val + 1; // 2

    const input_data = [_]f32{ 1, 2, 3, 4 };
    const weight_std = [_]f32{
        1, 1, 1, // oc=0: sum
        1, 0, 0, // oc=1: first only
    };
    var weight_repacked: [6]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    var output_data: [4]f32 = undefined;

    conv1d(out_channels, in_channels, out_width, in_width, kernel_size, stride_val, padding_val, &input_data, &weight_repacked, null, &output_data, .none);

    // ow=0: in_pos = 0*2+k-1 = {-1,0,1}. oc=0: 0+1+2=3. oc=1: 0=0 (pad) -> wait
    // ow=0: k=0 -> pos=-1 (pad), k=1 -> pos=0 (in[0]=1), k=2 -> pos=1 (in[1]=2)
    // oc=0: 0+1+2=3.  oc=1: 0+1+0=1 (w=[1,0,0], vals=[0,1,2])... no wait
    // oc=1 weight is [1,0,0]: k=0 padded(0), k=1 in[0]=1, k=2 in[1]=2
    // sum = 1*0 + 0*1 + 0*2 = 0. That's wrong. Let me recalculate.
    // oc=1 weight is [1,0,0]: w[0]*pad + w[1]*in[0] + w[2]*in[1] = 1*0 + 0*1 + 0*2 = 0

    // ow=1: in_pos = 1*2+k-1 = {1,2,3}. All valid.
    // oc=0: 2+3+4=9. oc=1: 1*2+0*3+0*4=2

    try testing.expectApproxEqAbs(output_data[0 * out_width + 0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[0 * out_width + 1], 9.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1 * out_width + 0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(output_data[1 * out_width + 1], 2.0, 1e-5);
}

test "conv1d VAD enc0 shape" {
    // Enc0: input [129, 4], weight [128, 129, 3], stride=1, padding=1 -> output [128, 4]
    const out_channels = 128;
    const in_channels = 129;
    const in_width = 4;
    const kernel_size = 3;
    const stride_val = 1;
    const padding_val = 1;
    const out_width = (in_width + 2 * padding_val - kernel_size) / stride_val + 1; // 4

    // Allocate with known patterns
    var input_data: [129 * 4]f32 = undefined;
    for (&input_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.8;
    }

    var weight_std: [128 * 129 * 3]f32 = undefined;
    for (&weight_std, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 13)) * 0.01 - 0.06;
    }

    var weight_repacked: [128 * 129 * 3]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    const bias_data: [128]f32 = [_]f32{0.1} ** 128;

    var output_simd: [128 * 4]f32 = undefined;
    var output_scalar: [128 * 4]f32 = undefined;

    // SIMD path
    conv1d(out_channels, in_channels, out_width, in_width, kernel_size, stride_val, padding_val, &input_data, &weight_repacked, &bias_data, &output_simd, .relu);

    // Scalar reference (using the same function — the tail handling covers everything)
    // Instead, compute reference manually
    for (0..out_channels) |oc| {
        for (0..out_width) |ow| {
            var sum_val: f32 = 0.0;
            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    const in_pos_signed: i64 = @as(i64, @intCast(ow * stride_val + k)) - @as(i64, @intCast(padding_val));
                    if (in_pos_signed >= 0 and in_pos_signed < @as(i64, @intCast(in_width))) {
                        const in_pos: usize = @intCast(in_pos_signed);
                        sum_val += input_data[ic * in_width + in_pos] * weight_std[oc * (in_channels * kernel_size) + ic * kernel_size + k];
                    }
                }
            }
            sum_val += bias_data[oc];
            sum_val = @max(sum_val, 0.0);
            output_scalar[oc * out_width + ow] = sum_val;
        }
    }

    // Compare
    for (0..out_channels * out_width) |i| {
        try testing.expectApproxEqAbs(output_simd[i], output_scalar[i], 1e-3);
    }
}

test "conv1d STFT shape" {
    // STFT: input [1, 640], weight [258, 1, 256], stride=128, padding=0 -> output [258, 4]
    const out_channels = 258;
    const in_channels = 1;
    const in_width = 640;
    const kernel_size = 256;
    const stride_val = 128;
    const out_width = (in_width - kernel_size) / stride_val + 1; // (640-256)/128+1 = 4

    var input_data: [640]f32 = undefined;
    for (&input_data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var weight_std: [258 * 1 * 256]f32 = undefined;
    for (&weight_std, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 19)) * 0.005 - 0.045;
    }

    var weight_repacked: [258 * 1 * 256]f32 = undefined;
    repackWeight(out_channels, in_channels, kernel_size, &weight_std, &weight_repacked);

    var output_simd: [258 * 4]f32 = undefined;
    var output_scalar: [258 * 4]f32 = undefined;

    // SIMD
    conv1d(out_channels, in_channels, out_width, in_width, kernel_size, stride_val, 0, &input_data, &weight_repacked, null, &output_simd, .none);

    // Scalar reference
    for (0..out_channels) |oc| {
        for (0..out_width) |ow| {
            var sum_val: f32 = 0.0;
            for (0..kernel_size) |k| {
                sum_val += input_data[ow * stride_val + k] * weight_std[oc * kernel_size + k];
            }
            output_scalar[oc * out_width + ow] = sum_val;
        }
    }

    for (0..out_channels * out_width) |i| {
        try testing.expectApproxEqAbs(output_simd[i], output_scalar[i], 1e-2);
    }
}
