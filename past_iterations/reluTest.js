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
      1, -2, 3, 4,
      -5, 6, -7, -8
    ]);
  
    const gpuBufferFirstMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
  
  
    // Compute shader code
  
    const shaderModule = device.createShaderModule({
      code: `
      struct Matrix {
        size : vec4<f32>; // batch_size , channel_size , height , width
        numbers: array<f32>,
    };
    
    
    @group(0) @binding(0) var<storage, read_write> inputImage : Matrix;
    
    
    @compute
@workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        // Guard against out-of-bounds work group sizes.
        if (global_id.x >= u32(inputImage.size.w) || global_id.y >= u32(inputImage.size.z) || global_id.z >= u32(inputImage.size.y)) {
            return;
        }
        let index = global_id.z * u32(inputImage.size.z) * u32(inputImage.size.w) + global_id.y * u32(inputImage.size.w) + global_id.x;
        var result = inputImage.numbers[index];
        if (result < 0.) {
            inputImage.numbers[index] = 0.2 * result; 
        }
        
        
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
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      gpuBufferFirstMatrix /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      firstMatrix.byteLength /* size */
    );
  
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);
  
  
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    console.log(new Float32Array(arrayBuffer));
  })();
  