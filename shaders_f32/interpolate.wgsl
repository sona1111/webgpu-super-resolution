[[block]] struct Matrix {
    numbers: array<f32>;
};

[[block]] struct UBO {
    inputSizesX: u32;
  inputSizesY: u32;
  inputSizesZ: u32;
};


[[group(0), binding(0)]] var<storage, read> inputImage : Matrix;
[[group(0), binding(1)]] var<storage, read_write> resultImage : Matrix;
[[group(0), binding(2)]] var<storage, read> ufs : UBO;


[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= ufs.inputSizesZ || global_id.y >= ufs.inputSizesY || global_id.z >= ufs.inputSizesX) {
        return;
    }

    //resultImage.size = vec4<f32>(inputImage.size.x, inputImage.size.y, 2.f * inputImage.size.z, 2.f * inputImage.size.w);
    let index = global_id.z * u32(ufs.inputSizesZ) * u32(ufs.inputSizesY) + global_id.y * u32(ufs.inputSizesZ) + global_id.x;
    var result = inputImage.numbers[index];
    let channel_area = 4u * ufs.inputSizesZ * ufs.inputSizesY;
    let result_width = 2u * ufs.inputSizesZ;
    let index1 = global_id.z * channel_area + global_id.y * 2u * result_width + 2u * global_id.x;
    let index2 = global_id.z * channel_area + global_id.y * 2u * result_width + 2u * global_id.x + 1u;
    let index3 = global_id.z * channel_area + (global_id.y * 2u + 1u)* result_width + 2u * global_id.x;
    let index4 = global_id.z * channel_area + (global_id.y * 2u + 1u)* result_width + 2u * global_id.x + 1u;
    resultImage.numbers[index1] = result; 
    resultImage.numbers[index2] = result; 
    resultImage.numbers[index3] = result; 
    resultImage.numbers[index4] = result; 
}
