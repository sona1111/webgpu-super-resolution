
[[block]] struct Array {

    numbers : array<f32>; 
};


[[block]] struct UBO {
  outputSizes: array<i32, 2>;
  inputSizes: array<i32, 3>;
  kernSizes: array<i32, 4>;
  offsetw: u32;
  offsetb: i32;
  inputOffset: u32;
  outputOffset: i32;
};

[[group(0), binding(0)]] var<storage, read> inputKernel : Array;
[[group(0), binding(1)]] var<storage, read> inputBias : Array;
[[group(0), binding(2)]] var<storage, read_write> resultImage : Array;
[[group(0), binding(3)]] var<storage, read> ufs : UBO;
[[group(0), binding(4)]] var<storage, read> inputImage : Array;



[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
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
    var kernel_area = u32(ufs.kernSizes[2]) * u32(ufs.kernSizes[3]);
    var image_area = u32(ufs.inputSizes[1]) * u32(ufs.inputSizes[2]);
    for (var z = 0; z < ufs.inputSizes[0]; z = z + 1) {
        for (var i = -1; i < 2; i = i + 1) {
            for (var j = -1; j < 2; j = j + 1) {

                if(y+i < 0 || x+j < 0 || y+i >= ufs.outputSizes[1] || x+j >= ufs.outputSizes[0]){
                    continue;
                }
                let kernIndex = outmost_dim +
                                u32(z) * kernel_area +
                                u32(i+1) * u32(ufs.kernSizes[3]) +
                                u32(j+1);
                let imageIndex = u32(z) * image_area +
                                 u32(y+i) * u32(ufs.inputSizes[1]) +
                                 u32(x+j);

                result = result + (inputKernel.numbers[ufs.offsetw + kernIndex] * inputImage.numbers[ufs.inputOffset + imageIndex]);


            }
        }
    }

    result = result + inputBias.numbers[ufs.offsetb + co];
    if (result < 0.) {
        result = 0.2 * result; 
    }



    let index = (co * ufs.outputSizes[0] * ufs.outputSizes[1]) + (ufs.outputSizes[0] * y) + x + ufs.outputOffset;

    resultImage.numbers[index] = result;




}
