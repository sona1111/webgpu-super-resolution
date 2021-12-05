// [[block]] struct Matrix {
//     size : vec4<f32>; // batch_size , channel_size , height , width
//     numbers: array<f32>;
// };

// [[block]] struct Array {
//     size : f32; // channel_size
//     numbers : array<f32>; 
// };

// [[group(0), binding(0)]] var<storage, read> inputImage : Matrix;
// [[group(0), binding(1)]] var<storage, write> resultImage : Matrix;


// [[stage(compute), workgroup_size(4, 4, 4)]]
// fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
//     // Guard against out-of-bounds work group sizes.
//     if (global_id.x >= u32(inputImage.size.w) || global_id.y >= u32(inputImage.size.z) || global_id.z >= u32(inputImage.size.y)) {
//         return;
//     }

//     resultImage.size = vec4<f32>(inputImage.size.x, inputImage.size.y, 2.f * inputImage.size.z, 2.f * inputImage.size.w);
//     let index = global_id.z * u32(inputImage.size.z) * u32(inputImage.size.w) + global_id.y * u32(inputImage.size.w) + global_id.x;
//     var result = inputImage.numbers[index];
//     let index1 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + global_id.y * 2u * u32(resultImage.size.w) + 2u * global_id.x;
//     let index2 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + global_id.y * 2u * u32(resultImage.size.w) + 2u * global_id.x + 1u;
//     let index3 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + (global_id.y * 2u + 1u)* u32(resultImage.size.w) + 2u * global_id.x;
//     let index4 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + (global_id.y * 2u + 1u)* u32(resultImage.size.w) + 2u * global_id.x + 1u;
//     resultImage.numbers[index1] = result; 
//     resultImage.numbers[index2] = result; 
//     resultImage.numbers[index3] = result; 
//     resultImage.numbers[index4] = result; 
// }

[[block]] struct Matrix {
    numbers: array<f32>;
};

[[block]] struct UBO {
  inputSizes: vec3<u32>; //channel_size , height , width
};


[[group(0), binding(0)]] var<storage, read> inputImage : Matrix;
[[group(0), binding(1)]] var<storage, write> resultImage : Matrix;
[[group(0), binding(2)]] var<storage, read> ufs : UBO;


[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= ufs.inputSizes.z || global_id.y >= ufs.inputSizes.y || global_id.z >= ufs.inputSizes.x) {
        return;
    }

    //resultImage.size = vec4<f32>(inputImage.size.x, inputImage.size.y, 2.f * inputImage.size.z, 2.f * inputImage.size.w);
    let index = global_id.z * u32(ufs.inputSizes.z) * u32(ufs.inputSizes.y) + global_id.y * u32(ufs.inputSizes.z) + global_id.x;
    var result = inputImage.numbers[index];
    let channel_area = 4u * ufs.inputSizes.z * ufs.inputSizes.y;
    let result_width = 2u * ufs.inputSizes.z;
    let index1 = global_id.z * channel_area + global_id.y * 2u * result_width + 2u * global_id.x;
    let index2 = global_id.z * channel_area + global_id.y * 2u * result_width + 2u * global_id.x + 1u;
    let index3 = global_id.z * channel_area + (global_id.y * 2u + 1u)* result_width + 2u * global_id.x;
    let index4 = global_id.z * channel_area + (global_id.y * 2u + 1u)* result_width + 2u * global_id.x + 1u;
    resultImage.numbers[index1] = result; 
    resultImage.numbers[index2] = result; 
    resultImage.numbers[index3] = result; 
    resultImage.numbers[index4] = result; 
}
