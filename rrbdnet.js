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

    const inp_img = getImgDataFromImgElem(document.getElementById('sm_img'));
    const modeldata = await getModelData("ESRGAN_py/RRDB_ESRGAN_x4");
    console.log(inp_img)

    function copy_mat_gpu(floatArr, arrclass, usage){
        // get GPU pointer for CPU float32 array and copy data to it
        const gpuArray = device.createBuffer({
            mappedAtCreation: true,
            size: floatArr.byteLength,
            usage: usage
        });
        const gpuArrayRNG = gpuArray.getMappedRange();
        new arrclass(gpuArrayRNG).set(floatArr);
        gpuArray.unmap();
        return gpuArray;
    }


    function get_mat_gpu_output(length){
        // get GPU pointer for an output array
        const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (length); // length + 4?
        const resultMatrixBuffer = device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        return resultMatrixBuffer
    }

    function vstack(arrs){
        // stack multiple arrs along first axis
        let total_first_dim = 0;
        for(let arr of arrs){
            total_first_dim += (arr.length / inp_img.w / inp_img.h);
        }
        const new_arr = new Float32Array(total_first_dim * inp_img.w * inp_img.h);

        let offset = 0;
        for(let arr of arrs){
            new_arr.set(arr, offset);
            offset += arr.length;
        }
        return new_arr;


    }

    async function gpuexec_conv(output, input, inputshape, weight, weightshape, bias, in_ch_idx, out_ch_idx, shaderModule) {

        // Pipeline setup

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
            ]
        });

        const computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            }
        });

        let aux_arr = [in_ch_idx, out_ch_idx, inp_img.w, inp_img.h].concat(inputshape).concat(weightshape)
        const channelIdxs = new Int32Array(aux_arr);

        const unifbuffer = copy_mat_gpu(channelIdxs, Int32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST


        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: input
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: weight
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: bias
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: output
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: unifbuffer
                    }
                }
            ]
        });

        // Compute shader code


        // Commands submission

        const commandEncoder = device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        passEncoder.dispatch(inp_img.w, inp_img.h);
        //passEncoder.dispatch(1, 1);
        //passEncoder.dispatch(1, 1, 1);
        passEncoder.endPass();

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);





    }

    async function gpuexec_readbuf(output, outputsize){
        const commandEncoder = device.createCommandEncoder();

        // Get a GPU buffer for reading in an unmapped state.
        const gpuReadBuffer = device.createBuffer({
            size: outputsize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Encode commands for copying buffer to buffer.
        commandEncoder.copyBufferToBuffer(
            output /* source buffer */,
            0 /* source offset */,
            gpuReadBuffer /* destination buffer */,
            0 /* destination offset */,
            outputsize /* size */
        );

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        // Read buffer.
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = gpuReadBuffer.getMappedRange();
        return new Float32Array(arrayBuffer);
    }

    async function conv_fwd(inp, w, wshape, b, bshape, relu){
        if(Array.isArray(inp)){  // not float32 array, standard JS array
            inp = vstack(inp);
        }
        console.log('output size', wshape[0] * inp_img.w * inp_img.h)
        const output =  get_mat_gpu_output(wshape[0] * inp_img.w * inp_img.h);
        const input = copy_mat_gpu(inp_img.c, Float32Array, GPUBufferUsage.STORAGE);
        const weight = copy_mat_gpu(w, Float32Array, GPUBufferUsage.STORAGE);
        const bias = copy_mat_gpu(b, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((wshape[0] * inp_img.w * inp_img.h));


        const shaderModuleConv = device.createShaderModule({
            code: await read_shader( 'conv2d_norelu.wgsl')
        });
        const shaderModuleAddBias = device.createShaderModule({
            code: await read_shader( 'addbias.wgsl')
        });
        //console.log(await read_shader( 'conv2d_norelu.wgsl'))

        console.log(output, input, weight, bias);
        const in_ch_count = inp.length / (inp_img.w * inp_img.h);
        for(let out_ch_idx=0; out_ch_idx<wshape[0]; out_ch_idx++){
            for(let in_ch_idx=0; in_ch_idx<in_ch_count; in_ch_idx++){
                if(relu){

                }else{
                    await gpuexec_conv(output, input, [in_ch_count, inp_img.w, inp_img.h],
                        weight, wshape, bias, in_ch_idx, out_ch_idx, shaderModuleConv)
                    //console.log(res);


                }
            }
            await gpuexec_conv(output, input, [in_ch_count, inp_img.w, inp_img.h],
                               weight, wshape, bias, 0, out_ch_idx, shaderModuleAddBias)

        }


        return await gpuexec_readbuf(output, outputsize);

    }

    async function esrgan(){


        async function eval_conv(name, inp, relu){
            console.log(modeldata[name])
            return await conv_fwd(
                inp,
                modeldata[name].w,
                modeldata[name].wshape,
                modeldata[name].b,
                modeldata[name].bshape,
                relu
            );
        }

        let fea = await eval_conv('conv_first', inp_img.c, false);
        console.log(fea);
        return fea;

    }
    await esrgan();
    return;
  
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
  
    const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (firstMatrix[0] * secondMatrix[0] * firstMatrix[2] * firstMatrix[3]);
    const resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
  
  

  
    async function exec_on_gpu(second_matrix) {
        // a very simple / naive way to take a kernel and execute it

        const shaderModule = device.createShaderModule({
            code: await read_shader( 'set_sim.wgsl')
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                // {
                //     binding: 4,
                //     visibility: GPUShaderStage.COMPUTE,
                //     buffer: {
                //         type: "read-only-storage"
                //     }
                // },
            ]
        });

        // Bind group
        const computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            }
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
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
  