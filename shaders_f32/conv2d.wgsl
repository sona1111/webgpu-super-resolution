struct Matrix {
    size : vec4<f32>; // batch_size , channel_size , height , width
    numbers: array<f32>,
};

struct Array {
    size : f32; // channel_size
    numbers : array<f32>,
};

@group(0) @binding(0) var<storage, read> inputImage : Matrix;
@group(0) @binding(1) var<storage, read> inputKernel : Matrix;
@group(0) @binding(2) var<storage, read> inputBias : Array;
@group(0) @binding(3) var<storage, read_write> resultImage : Matrix;


@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= u32(inputImage.size.w) || global_id.y >= u32(inputImage.size.z) || global_id.z >= u32(inputKernel.size.x)) {
        return;
    }

    resultImage.size = vec4<f32>(inputImage.size.x, inputKernel.size.x, inputImage.size.z, inputImage.size.w);
    
    var result = 0.0;
    for (var i = 0; i < i32(inputKernel.size.y); i = i + 1) {
        for (var j = 0; j < i32(inputKernel.size.z); j = j + 1) {
            for (var k = 0; k < i32(inputKernel.size.w); k = k + 1) {
                let imageX = i32(global_id.x) - i32(inputKernel.size.w) / 2 + k;
                let imageY = i32(global_id.y) - i32(inputKernel.size.z) / 2 + j;
                if (imageX >= 0 && imageX < i32(inputImage.size.w) && imageY >= 0 && imageY < i32(inputImage.size.z)) {
                    let kernIndex = global_id.z * u32(inputKernel.size.y) * u32(inputKernel.size.z) * u32(inputKernel.size.w) + 
                                        u32(i) * u32(inputKernel.size.z) * u32(inputKernel.size.w) + u32(j) * u32(inputKernel.size.w) + u32(k);
                    let imageIndex = u32(i) * u32(inputImage.size.z) * u32(inputImage.size.w) + u32(imageY) * u32(inputImage.size.w) + u32(imageX);
                    result = result + inputImage.numbers[imageIndex] * inputKernel.numbers[kernIndex];
                }

            }
        }
    }
    result = result + inputBias.numbers[global_id.z];
    let index = global_id.z * u32(inputImage.size.z) * u32(inputImage.size.w) + global_id.y * u32(inputImage.size.w) + global_id.x;
    resultImage.numbers[index] = result;
}
