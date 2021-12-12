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
      1, 1, 2, 2,
      1, 2, 3, 4,
    ]);
  
    const gpuBufferFirstMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
  
  
    // Second Matrix
  
    const secondMatrix = new Float32Array([
      2, 1, 3, 3,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17, 18
    ]);
  
    const gpuBufferSecondMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: secondMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();
  

    const biasArray = new Float32Array([
        2,
        0.3, 0.6
      ]);
    
      const gpuBufferBiasArray = device.createBuffer({
        mappedAtCreation: true,
        size: biasArray.byteLength,
        usage: GPUBufferUsage.STORAGE
      });
      const arrayBufferBiasArray = gpuBufferBiasArray.getMappedRange();
      new Float32Array(arrayBufferBiasArray).set(biasArray);
      gpuBufferBiasArray.unmap();
    
    // Result Matrix
  
    const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (4 + firstMatrix[0] * secondMatrix[0] * firstMatrix[2] * firstMatrix[3]);
    const resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
  
  
    // Compute shader code
  
    const shaderModule = device.createShaderModule({
      code: `
      [[block]] struct Matrix {
        size : vec4<f32>; // batch_size , channel_size , height , width
        numbers: array<f32>;
    };
    
    [[block]] struct Array {
        size : f32; // channel_size
        numbers : array<f32>; 
    };
    
    [[group(0), binding(0)]] var<storage, read> inputImage : Matrix;
    [[group(0), binding(1)]] var<storage, read> inputKernel : Matrix;
    [[group(0), binding(2)]] var<storage, read> inputBias : Array;
    [[group(0), binding(3)]] var<storage, write> resultImage : Matrix;
    
    
    [[stage(compute), workgroup_size(4, 4, 4)]]
    fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
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
            buffer: gpuBufferSecondMatrix
          }
        },
        {
            binding: 2,
            resource: {
              buffer: gpuBufferBiasArray
            }
          },
        {
          binding: 3,
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
    const z = Math.ceil(secondMatrix[0] / 4); 
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
  