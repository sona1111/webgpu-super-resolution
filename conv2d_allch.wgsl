
[[block]] struct Array {

    numbers : array<f32>; 
};


[[block]] struct UBO {
  channelInputs: array<i32, 1>;
  outputSizes: array<i32, 2>;
  inputSizes: array<i32, 3>;
  kernSizes: array<i32, 4>;
};

[[group(0), binding(0)]] var<storage, read> inputImage : Array;
[[group(0), binding(1)]] var<storage, read> inputKernel : Array;
[[group(0), binding(2)]] var<storage, read> inputBias : Array;
[[group(0), binding(3)]] var<storage, write> resultImage : Array;
[[group(0), binding(4)]] var<storage, read> ufs : UBO;



[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes.

    if (global_id.x >= u32(ufs.outputSizes[0]) || global_id.y >= u32(ufs.outputSizes[1]) || global_id.z >= u32(ufs.kernSizes[0])) {
        return;
    }



    //resultImage.size = vec4<f32>(inputImage.size.x, inputKernel.size.x, inputImage.size.z, inputImage.size.w);
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let co = i32(global_id.z);
    let z = ufs.channelInputs[0];



    //for (var co = 0; co < ufs.kernSizes[0]; co = co + 1) { //ufs.kernSizes[0]

    var result = 0.0;
    for (var i = -1; i < 2; i = i + 1) {
        for (var j = -1; j < 2; j = j + 1) {

            if(y+i < 0 || x+j < 0 || y+i >= ufs.outputSizes[1] || x+j >= ufs.outputSizes[0]){
                continue;
            }
            let kernIndex = u32(co) * u32(ufs.kernSizes[1]) * u32(ufs.kernSizes[2]) * u32(ufs.kernSizes[3]) +
                            u32(z) * u32(ufs.kernSizes[2]) * u32(ufs.kernSizes[3]) +
                            u32(i+1) * u32(ufs.kernSizes[3]) +
                            u32(j+1);
            let imageIndex = u32(z) * u32(ufs.inputSizes[1]) * u32(ufs.inputSizes[2]) +
                             u32(y+i) * u32(ufs.inputSizes[1]) +
                             u32(x+j);
            //resultImage.numbers[dbg_idx] = f32(inputImage.numbers[imageIndex]);
            result = result + (inputKernel.numbers[kernIndex] * inputImage.numbers[imageIndex]);


        }
    }

    if(z == 0){
        result = result + inputBias.numbers[co];
    }


    let index = (co * ufs.outputSizes[0] * ufs.outputSizes[1]) + (ufs.outputSizes[0] * y) + x;

    resultImage.numbers[index] = resultImage.numbers[index] + result;




}
