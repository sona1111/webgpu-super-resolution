
struct Array {

    numbers : array<f32>,
};


struct UBO {
  channelIdxs: array<i32, 2>,
  outputSizes: array<i32, 2>,
  inputSizes: array<i32, 3>,
  kernSizes: array<i32, 4>,
};

@group(0) @binding(0) var<storage, read> inputImage : Array;
@group(0) @binding(1) var<storage, read> inputKernel : Array;
@group(0) @binding(2) var<storage, read> inputBias : Array;
[[group(0), binding(3)]] var<storage, read_write> resultImage : Array;
[[group(0), binding(4)]] var<storage, read> ufs : UBO;



[[stage(compute), workgroup_size(4, 4)]]
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    // Guard against out-of-bounds work group sizes.

    if (global_id.x >= u32(ufs.outputSizes[0]) || global_id.y >= u32(ufs.outputSizes[1])) {
        return;
    }

    //resultImage.size = vec4<f32>(inputImage.size.x, inputKernel.size.x, inputImage.size.z, inputImage.size.w);
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let index = (ufs.channelIdxs[1] * ufs.outputSizes[0] * ufs.outputSizes[1]) + (ufs.outputSizes[0] * y) + x;

    resultImage.numbers[index] = resultImage.numbers[index] + inputBias.numbers[ufs.channelIdxs[1]];

}
