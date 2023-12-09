
struct Matrix {
    numbers: array<f32>,
};

struct UBO {
    inputSizesX: u32,
  inputSizesY: u32,
  inputSizesZ: u32,
  inputOffset: u32,
};


@group(0) @binding(0) var<storage, read_write> inputImage : Matrix;
@group(0) @binding(1) var<storage, read> ufs : UBO;


@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= ufs.inputSizesZ || global_id.y >= ufs.inputSizesY || global_id.z >= ufs.inputSizesX) {
        return;
    }
    let index = global_id.z * ufs.inputSizesY * ufs.inputSizesZ + global_id.y * ufs.inputSizesZ + global_id.x + ufs.inputOffset;
    var result = inputImage.numbers[index];
    if (result < 0.) {
        inputImage.numbers[index] = 0.2 * result; 
    }
    
    
}
