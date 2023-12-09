
struct Array {

    numbers : array<f32>,
};


struct UBO {
  outputSizes: array<i32, 2>,
  inputSizes: array<i32, 3>,
  kernSizes: array<i32, 4>,
  offsetw: u32,
  offsetb: i32,
  inputOffset: u32,
  outputOffset: i32,
};

@group(0) @binding(0) var<storage, read> inputKernel : Array;
@group(0) @binding(1) var<storage, read> inputBias : Array;
@group(0) @binding(2) var<storage, read_write> resultImage : Array;
@group(0) @binding(3) var<storage, read> ufs : UBO;



@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    // Guard against out-of-bounds work group sizes.

    if (global_id.x >= u32(ufs.outputSizes[0]) || global_id.y >= u32(ufs.outputSizes[1]) || global_id.z >= u32(ufs.kernSizes[0])) {
        return;
    }



    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let co = i32(global_id.z);



    //for (var co = 0; co < ufs.kernSizes[0]; co = co + 1) { //ufs.kernSizes[0]

    var result = 0.0;
    var outmost_dim = u32(co) * u32(ufs.kernSizes[1]) * u32(ufs.kernSizes[2]) * u32(ufs.kernSizes[3]);
    outmost_dim = outmost_dim + ufs.offsetw;
    var image_area = u32(ufs.inputSizes[1]) * u32(ufs.inputSizes[2]);
    for (var z = 0u; z < u32(ufs.inputSizes[0]); z = z + 1u) {
        if(y - 1 >= 0 && x - 1 >= 0){
            let kernIndex = outmost_dim + z * 9u;
            let imageIndex = z * image_area + u32(y+-1) * u32(ufs.inputSizes[1]) + u32(x+-1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(y - 1 >= 0){
            let kernIndex = outmost_dim + z * 9u + 1u;
            let imageIndex = z * image_area + u32(y+-1) * u32(ufs.inputSizes[1]) + u32(x);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(y - 1 >= 0 && x + 1 < ufs.outputSizes[0]){
            let kernIndex = outmost_dim + z * 9u + 2u;
            let imageIndex = z * image_area + u32(y+-1) * u32(ufs.inputSizes[1]) + u32(x + 1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(x - 1 >= 0){
            let kernIndex = outmost_dim + z * 9u + 3u;
            let imageIndex = z * image_area + u32(y) * u32(ufs.inputSizes[1]) + u32(x+-1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        let kernIndex = outmost_dim + z * 9u + 4u;
        let imageIndex = z * image_area + u32(y) * u32(ufs.inputSizes[1]) + u32(x);
        result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        if(x + 1 < ufs.outputSizes[0]){
            let kernIndex = outmost_dim + z * 9u + 5u;
            let imageIndex = z * image_area + u32(y) * u32(ufs.inputSizes[1]) + u32(x + 1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(y + 1 < ufs.outputSizes[1] && x - 1 >= 0){
            let kernIndex = outmost_dim + z * 9u + 6u;
            let imageIndex = z * image_area + u32(y+1) * u32(ufs.inputSizes[1]) + u32(x+-1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(y + 1 < ufs.outputSizes[1]){
            let kernIndex = outmost_dim + z * 9u + 7u;
            let imageIndex = z * image_area + u32(y+1) * u32(ufs.inputSizes[1]) + u32(x);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
        if(y + 1 < ufs.outputSizes[1] && x + 1 < ufs.outputSizes[0]){
            let kernIndex = outmost_dim + z * 9u + 8u;
            let imageIndex = z * image_area + u32(y+1) * u32(ufs.inputSizes[1]) + u32(x + 1);
            result = result + (inputKernel.numbers[kernIndex] * resultImage.numbers[ufs.inputOffset + imageIndex]);
        }
    }
    result = result + inputBias.numbers[ufs.offsetb + co];
    if (result < 0.) {
        result = result * 0.2;
    }


    let index = (co * ufs.outputSizes[0] * ufs.outputSizes[1]) + (ufs.outputSizes[0] * y) + x + ufs.outputOffset;

    resultImage.numbers[index] = result;




}
