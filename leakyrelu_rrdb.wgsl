
[[block]] struct Matrix {
    numbers: array<f32>;
};

[[block]] struct UBO {
  inputSizes: vec3<u32>; //channel_size , height , width
  inputOffset: u32;
};


[[group(0), binding(0)]] var<storage, read_write> inputImage : Matrix;
[[group(0), binding(1)]] var<storage, read> ufs : UBO;


[[stage(compute), workgroup_size(4, 4, 4)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= ufs.inputSizes.z || global_id.y >= ufs.inputSizes.y || global_id.z >= ufs.inputSizes.x) {
        return;
    }
    let index = global_id.z * ufs.inputSizes.y * ufs.inputSizes.z + global_id.y * ufs.inputSizes.z + global_id.x + ufs.inputOffset;
    var result = inputImage.numbers[index];
    if (result < 0.) {
        inputImage.numbers[index] = 0.2 * result; 
    }
    
    
}
