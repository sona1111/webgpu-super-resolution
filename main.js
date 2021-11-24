// const vertexShaderWgslCode =
//     `
//         const pos : array<vec2<f32>, 3> = array<vec2<f32>, 3>(
//         vec2<f32>(0.0, 0.5),
//         vec2<f32>(-0.5, -0.5),
//         vec2<f32>(0.5, -0.5));
//
//         [[builtin(position)]] var<out> Position : vec4<f32>;
//         [[builtin(vertex_idx)]] var<in> VertexIndex : i32;
//
//         [[stage(vertex)]]
//         fn main() -> void {
//             Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
//             return;
//         }
//     `;
//
//
// async function main(){
//     const [m, n, k] = [64, 64, 64];
//     const array_a = new Float32Array(m * k);//m*k row-major matrix
//     const array_b = new Float32Array(k * n);//k*n row-major matrix
//     // fill array_a, array_b
//     for (let i = 0; i < array_a.length; i++) {
//         array_a[i] = Math.random();
//     }
//     for (let i = 0; i < array_b.length; i++) {
//         array_b[i] = Math.random();
//     }
//     const alpha = 1.0;
//     const result = await webgpublas.sgemm(m, n, k, alpha, array_a, array_b);
//     console.log(result); // m*n row-major matrix (Float32Array)
// }

//const modelWeights = await getModelWeights("ESRGAN/small");
//console.log(modelWeights);

function getImgDataFromImgElem(imgElem){
    const canvas = document.createElement('canvas');
    console.log(imgElem.width)
    canvas.width = imgElem.width;
    canvas.height = imgElem.height;
    canvas.getContext('2d').drawImage(imgElem, 0, 0, imgElem.width, imgElem.height);
    const pixelData = canvas.getContext('2d').getImageData(0, 0, imgElem.width, imgElem.height).data;
    const r = new Float32Array(pixelData.length/4);
    const g = new Float32Array(pixelData.length/4);
    const b = new Float32Array(pixelData.length/4);
    let j=0;
    for(let i=0;i<pixelData.length;i+=4){
        r[j]=pixelData[i];
        g[j]=pixelData[i+1];
        b[j]=pixelData[i+2];
        j++;
    }
    return [r, g, b];
}

async function main(){
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { return; }
    const device = await adapter.requestDevice();

    // Get a GPU buffer in a mapped state and an arrayBuffer for writing.
    const gpuBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: 4,
        usage: GPUBufferUsage.MAP_WRITE
    });
    const arrayBuffer = gpuBuffer.getMappedRange();

    // Write bytes to buffer.
    new Uint8Array(arrayBuffer).set([0, 1, 2, 3]);
}

main();
console.log(getImgDataFromImgElem(document.getElementById('sm_img')));
