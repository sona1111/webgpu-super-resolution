// [[block]] struct Matrix {
//     size : vec4<f32>; // batch_size , channel_size , height , width
//     numbers: array<f32>;
// };


// [[group(0), binding(0)]] var<storage, read_write> inputImage : Matrix;


// [[stage(compute), workgroup_size(4, 4, 4)]]
// fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
//     // Guard against out-of-bounds work group sizes.
//     if (global_id.x >= u32(inputImage.size.w) || global_id.y >= u32(inputImage.size.z) || global_id.z >= u32(inputImage.size.y)) {
//         return;
//     }
//     let index = global_id.z * u32(inputImage.size.z) * u32(inputImage.size.w) + global_id.y * u32(inputImage.size.w) + global_id.x;
//     var result = inputImage.numbers[index];
//     if (result < 0.f) {
//         inputImage.numbers[index] = 0.2 * result; 
//     }
    
    
// }
[[block]] struct Matrix {
    numbers: array<f32>;
};

[[block]] struct UBO {
    inputSizesX: u32;
  inputSizesY: u32;
  inputSizesZ: u32;
};


[[group(0), binding(0)]] var<storage, read> inputImage : Matrix;
[[group(0), binding(1)]] var<storage, read_write> outputImage : Matrix;
[[group(0), binding(2)]] var<storage, read> ufs : UBO;


[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= ufs.inputSizesZ || global_id.y >= ufs.inputSizesY || global_id.z >= ufs.inputSizesX) {
        return;
    }
    let index = global_id.z * ufs.inputSizesY * ufs.inputSizesZ + global_id.y * ufs.inputSizesZ + global_id.x;
    var in = inputImage.numbers[index];
    var out = outputImage.numbers[index];
    outputImage.numbers[index] = 0.2 * out + in;
    
    
}
