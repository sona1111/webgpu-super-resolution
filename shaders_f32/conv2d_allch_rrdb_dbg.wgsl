
struct Array {

    numbers : array<f32>,
};


@group(0) @binding(0) var<storage, read_write> resultImage : Array;



@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3u) {

    resultImage.numbers[0] = 1.0;


}
