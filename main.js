const vertexShaderWgslCode =
    `
        const pos : array<vec2<f32>, 3> = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5));
      
        [[builtin(position)]] var<out> Position : vec4<f32>;
        [[builtin(vertex_idx)]] var<in> VertexIndex : i32;
      
        [[stage(vertex)]]
        fn main() -> void {
            Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
            return;
        }
    `;


async function main(){
    const [m, n, k] = [64, 64, 64];
    const array_a = new Float32Array(m * k);//m*k row-major matrix
    const array_b = new Float32Array(k * n);//k*n row-major matrix
    // fill array_a, array_b
    for (let i = 0; i < array_a.length; i++) {
        array_a[i] = Math.random();
    }
    for (let i = 0; i < array_b.length; i++) {
        array_b[i] = Math.random();
    }
    const alpha = 1.0;
    const result = await webgpublas.sgemm(m, n, k, alpha, array_a, array_b);
    console.log(result); // m*n row-major matrix (Float32Array)
}

//main();
