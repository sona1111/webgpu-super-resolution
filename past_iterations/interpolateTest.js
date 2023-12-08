(async () => {
    if (!navigator.gpu) {
      console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
      return;
    }
  
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log("Failed to get GPU adapter.");
      return;
    }
    const device = await adapter.requestDevice();
  
    // First Matrix
  
    const firstMatrix = new Float32Array([
      1, 2, 2, 2,
      1, 2, 3, 4,
      5, 6, 7, 8
    ]);
  
    const gpuBufferFirstMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
  
  
    
    // Result Matrix
  
    const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (4 + 4 * firstMatrix[0] * firstMatrix[1] * firstMatrix[2] * firstMatrix[3]);
    const resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
  
  
    // Compute shader code
  
    const shaderModule = device.createShaderModule({
      code: `
      struct Matrix {
        size : vec4<f32>; // batch_size , channel_size , height , width
        numbers: array<f32>,
    };
    
    struct Array {
        size : f32; // channel_size
        numbers : array<f32>,
    };
    
    @group(0) @binding(0) var<storage, read> inputImage : Matrix;
    @group(0) @binding(1) var<storage, write> resultImage : Matrix;
    
    
    @compute
@workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        // Guard against out-of-bounds work group sizes.
        if (global_id.x >= u32(inputImage.size.w) || global_id.y >= u32(inputImage.size.z) || global_id.z >= u32(inputImage.size.y)) {
            return;
        }
    
        resultImage.size = vec4<f32>(inputImage.size.x, inputImage.size.y, 2.f * inputImage.size.z, 2.f * inputImage.size.w);
        let index = global_id.z * u32(inputImage.size.z) * u32(inputImage.size.w) + global_id.y * u32(inputImage.size.w) + global_id.x;
        var result = inputImage.numbers[index];
        let index1 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + global_id.y * 2u * u32(resultImage.size.w) + 2u * global_id.x;
        let index2 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + global_id.y * 2u * u32(resultImage.size.w) + 2u * global_id.x + 1u;
        let index3 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + (global_id.y * 2u + 1u)* u32(resultImage.size.w) + 2u * global_id.x;
        let index4 = global_id.z * u32(resultImage.size.z) * u32(resultImage.size.w) + (global_id.y * 2u + 1u)* u32(resultImage.size.w) + 2u * global_id.x + 1u;
        resultImage.numbers[index1] = result; 
        resultImage.numbers[index2] = result; 
        resultImage.numbers[index3] = result; 
        resultImage.numbers[index4] = result; 
    }
    
      `
    });
    
    // Pipeline setup
    
    const computePipeline = device.createComputePipeline({
      compute: {
        module: shaderModule,
        entryPoint: "main"
      }
    });
  
  
    // Bind group
  
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0 /* index */),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: gpuBufferFirstMatrix
          }
        },
        {
          binding: 1,
          resource: {
            buffer: resultMatrixBuffer
          }
        }
      ]
    });
    
  
    // Commands submission
  
    const commandEncoder = device.createCommandEncoder();
  
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const x = Math.ceil(firstMatrix[3] / 4); // X dimension of the grid of workgroups to dispatch.
    const y = Math.ceil(firstMatrix[2] / 4); // Y dimension of the grid of workgroups to dispatch.
    const z = Math.ceil(firstMatrix[1] / 4); 
    passEncoder.dispatch(x, y, z);
    passEncoder.endPass();
  
    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      resultMatrixBufferSize /* size */
    );
  
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);
  
  
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    console.log(new Float32Array(arrayBuffer));
  })();
  