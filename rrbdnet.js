async function read_shader(path){
    const conv2dc = await fetch(path);
    return await conv2dc.text();
}

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

    // third matrix (just for testing)
    const thirdMatrix = new Float32Array([
        9, 1, 3, 3,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 77, 15, 16, 17, 18
    ]);

    const gpuBufferThirdMatrix = device.createBuffer({
        mappedAtCreation: true,
        size: secondMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferThirdMatrix = gpuBufferThirdMatrix.getMappedRange();
    new Float32Array(arrayBufferThirdMatrix).set(thirdMatrix);
    gpuBufferThirdMatrix.unmap();
  

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
      code: await read_shader('conv2d.wgsl')
    });
    
    // Pipeline setup
    
    const computePipeline = device.createComputePipeline({
      compute: {
        module: shaderModule,
        entryPoint: "main"
      }
    });
  
    async function exec_on_gpu(second_matrix) {


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
                        buffer: second_matrix
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
    }

    await exec_on_gpu(gpuBufferSecondMatrix);
    await exec_on_gpu(gpuBufferThirdMatrix);
  })();
  