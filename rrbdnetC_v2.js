async function read_shader(path){
    const conv2dc = await fetch(path);
    return await conv2dc.text();
}





(async () => {
    if (!navigator.gpu) {
      console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
      return;
    }
  
    //const adapter = await navigator.gpu.requestAdapter();
    const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
    if (!adapter) {
      console.log("Failed to get GPU adapter.");
      return;
    }
    const device = await adapter.requestDevice();

    const inp_img = getImgDataFromImgElem(document.getElementById('sm_img'));
    imagedata2Canvas(inp_img.c, document.getElementById('result'), inp_img.w, inp_img.h);
    const modeldata = await getModelData("ESRGAN_py/RRDB_ESRGAN_x4");


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

    const shaderModuleConv = device.createShaderModule({
        code: await read_shader( 'conv2d_allch.wgsl')
    });
    const shaderModuleReLU = device.createShaderModule({
        code: await read_shader( 'leakyrelu.wgsl')
    });
    const shaderModuleScaleRes = device.createShaderModule({
        code: await read_shader( 'scaleandresidual.wgsl')
    });
    const shaderModuleInterpolate = device.createShaderModule({
        code: await read_shader( 'interpolate.wgsl')
    });
    const shaderModuleAddition = device.createShaderModule({
        code: await read_shader( 'addition.wgsl')
    });

    async function gpuexec_conv(output, input, inputshape, weight, weightshape, bias, shaderModule) {

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

        let aux_arr = [inputshape[1], inputshape[2]].concat(inputshape).concat(weightshape)
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

        passEncoder.dispatch(inputshape[1], inputshape[2], weightshape[0]);
        //passEncoder.dispatch(1, 1);
        //passEncoder.dispatch(1, 1, 1);*
        passEncoder.endPass();

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);





    }

    

    async function gpuexec_relu(input, inputSize, shaderModule) {
        const computePipeline = device.createComputePipeline({
            compute: {
              module: shaderModule,
              entryPoint: "main"
            }
          });        
        
          // Bind group
          const matrixSize = new Uint32Array(inputSize);
  
          const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST
        
          const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
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
                    buffer: unifbuffer
                }
              }
            ]
          });
        const commandEncoder = device.createCommandEncoder();
  
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const x = Math.ceil(matrixSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(matrixSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(matrixSize[0] / 4); 
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
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

    async function scale_residual(input, output) {
        inputSize = [input.length / (inp_img.w * inp_img.h), inp_img.h, inp_img.w];
        const outputBuff =  copy_mat_gpu(output, Float32Array, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const inputBuff = copy_mat_gpu(input, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((inputSize[0] * inp_img.w * inp_img.h));


        const computePipeline = device.createComputePipeline({
            compute: {
              module: shaderModuleScaleRes,
              entryPoint: "main"
            }
          });        
        
          // Bind group
          const matrixSize = new Uint32Array(inputSize);
  
          const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST
        
          const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: inputBuff
                }
              },
              {
                binding: 1,
                resource: {
                    buffer: outputBuff
                }
              },
              {
                binding: 2,
                resource: {
                    buffer: unifbuffer
                }
              }
            ]
          });
        const commandEncoder = device.createCommandEncoder();
  
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const x = Math.ceil(matrixSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(matrixSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(matrixSize[0] / 4); 
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);
        const res =  await gpuexec_readbuf(outputBuff, outputsize);
        outputBuff.destroy();
        inputBuff.destroy();
        return res;
    }

    async function matrix_addition(input, output) {
        inputSize = [input.length / (inp_img.w * inp_img.h), inp_img.h, inp_img.w];
        const outputBuff =  copy_mat_gpu(output, Float32Array, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const inputBuff = copy_mat_gpu(input, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((inputSize[0] * inp_img.w * inp_img.h));


        const computePipeline = device.createComputePipeline({
            compute: {
              module: shaderModuleAddition,
              entryPoint: "main"
            }
          });        
        
          // Bind group
          const matrixSize = new Uint32Array(inputSize);
  
          const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST
        
          const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: inputBuff
                }
              },
              {
                binding: 1,
                resource: {
                    buffer: outputBuff
                }
              },
              {
                binding: 2,
                resource: {
                    buffer: unifbuffer
                }
              }
            ]
          });
        const commandEncoder = device.createCommandEncoder();
  
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const x = Math.ceil(matrixSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(matrixSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(matrixSize[0] / 4); 
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);
        const res = await gpuexec_readbuf(outputBuff, outputsize);
        outputBuff.destroy();
        inputBuff.destroy();
        return res;
    }

    async function up_resolution(input, input_shape) {
        inputSize = [input.length / (input_shape[0] * input_shape[1]), input_shape[0], input_shape[1]];
        const outputBuff =  get_mat_gpu_output(4 * inputSize[0] * input_shape[0] * input_shape[1]);
        const inputBuff = copy_mat_gpu(input, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((4 * inputSize[0] * input_shape[0] * input_shape[1]));


        const computePipeline = device.createComputePipeline({
            compute: {
              module: shaderModuleInterpolate,
              entryPoint: "main"
            }
          });        
        
          // Bind group
          const matrixSize = new Uint32Array(inputSize);
  
          const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST
        
          const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: inputBuff
                }
              },
              {
                binding: 1,
                resource: {
                    buffer: outputBuff
                }
              },
              {
                binding: 2,
                resource: {
                    buffer: unifbuffer
                }
              }
            ]
          });
        const commandEncoder = device.createCommandEncoder();
  
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const x = Math.ceil(matrixSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(matrixSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(matrixSize[0] / 4); 
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);
        const res = await gpuexec_readbuf(outputBuff, outputsize);
        outputBuff.destroy();
        inputBuff.destroy();
        return res;
    }



    async function conv_fwd(inp, inp_shape, w, wshape, b, bshape, relu){
        if(Array.isArray(inp)){  // not float32 array, standard JS array
            inp = vstack(inp);
        }
        //console.log('output size', wshape[0] * inp_shape[0] * inp_shape[1]) // Need to change this to be more generic
        const output =  get_mat_gpu_output(wshape[0] * inp_shape[0] * inp_shape[1]);
        const input = copy_mat_gpu(inp, Float32Array, GPUBufferUsage.STORAGE);
        const weight = copy_mat_gpu(w, Float32Array, GPUBufferUsage.STORAGE);
        const bias = copy_mat_gpu(b, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((wshape[0] * inp_shape[0] * inp_shape[1]));

        //console.log(await read_shader( 'conv2d_norelu.wgsl'))

        //console.log(output, input, weight, bias);
        const in_ch_count = inp.length / (inp_shape[0] * inp_shape[1]);

        await gpuexec_conv(output, input, [in_ch_count, inp_shape[1], inp_shape[0]],
            weight, wshape, bias, shaderModuleConv);

        if (relu) {
            await gpuexec_relu(output, [wshape[0], inp_shape[0], inp_shape[1]], shaderModuleReLU)
        }


        const res = await gpuexec_readbuf(output, outputsize);
        output.destroy();
        input.destroy();
        weight.destroy();
        bias.destroy();
        return res;

    }

    async function esrgan(){


        async function eval_conv(name, inp, inp_shape, relu){
            //console.log(modeldata[name])
            console.log(name);
            return await conv_fwd(
                inp,
                inp_shape,
                modeldata[name].w,
                modeldata[name].wshape,
                modeldata[name].b,
                modeldata[name].bshape,
                relu
            );
        }

        let fea = await eval_conv('conv_first', inp_img.c, [inp_img.h, inp_img.w], false);

        let rrdb_in = fea;
        //console.log(fea)
        for (let rrdb_chunk = 0; rrdb_chunk < 23; rrdb_chunk++) {
            let rdb_in = rrdb_in;
            for (let rdb = 1; rdb <= 3; rdb ++) {
                let rdb_x1 = await eval_conv('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv1', rdb_in, [inp_img.h, inp_img.w], true)
                // console.log(rdb_x1)
                let rbd_x2 = await eval_conv('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv2', vstack([rdb_in, rdb_x1]), [inp_img.h, inp_img.w], true)
                // console.log(rbd_x2)
                let rbd_x3 = await eval_conv('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv3', vstack([rdb_in, rdb_x1, rbd_x2]), [inp_img.h, inp_img.w], true)
                let rbd_x4 = await eval_conv('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv4', vstack([rdb_in, rdb_x1, rbd_x2, rbd_x3]), [inp_img.h, inp_img.w], true)
                let rbd_x5 = await eval_conv('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv5', vstack([rdb_in, rdb_x1, rbd_x2, rbd_x3, rbd_x4]), [inp_img.h, inp_img.w], false)
                rdb_in = await scale_residual(rdb_in, rbd_x5);
                // console.log(rdb_in)
            }
            rrdb_in = await scale_residual(rrdb_in, rdb_in)
            // console.log(rrdb_in)
        }
        const trunk = await eval_conv('trunk_conv', rrdb_in, [inp_img.h, inp_img.w], false);
        fea = await matrix_addition(trunk, fea);
        // console.log(fea)
        fea = await up_resolution(fea, [inp_img.h, inp_img.w]);
        fea = await eval_conv('upconv1', fea, [2 * inp_img.h, 2 * inp_img.w], true);
        fea = await up_resolution(fea, [2 * inp_img.h, 2 * inp_img.w]);
        fea = await eval_conv('upconv2', fea, [4 * inp_img.h, 4 * inp_img.w], true);
        fea = await eval_conv('HRconv', fea, [4 * inp_img.h, 4 * inp_img.w], true);
        fea = await eval_conv('conv_last', fea, [4 * inp_img.h, 4 * inp_img.w], false);
        // console.log(fea)
        return fea;

    }
    var startTime = performance.now()

    const nn_result = await esrgan();
    console.log(nn_result)

    var endTime = performance.now()

    console.log(`NN run took ${endTime - startTime} milliseconds`)

    imagedata2Canvas(nn_result, document.getElementById('result'), inp_img.w*4, inp_img.h*4);


  })();
  