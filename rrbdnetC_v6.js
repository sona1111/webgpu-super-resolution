async function read_shader(path){
    const conv2dc = await fetch(path);
    return await conv2dc.text();
}

async function run_nn(input_elem, output_elem, _modeldata){

    document.getElementById('imageUpload').disabled = true;
    if (!navigator.gpu) {
      console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
      return;
    }
  
    //const adapter = await navigator.gpu.requestAdapter();
    //Chooses the high performance GPU instead of the integrated GPU
    const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
    if (!adapter) {
      console.log("Failed to get GPU adapter.");
      return;
    }
    const device = await adapter.requestDevice();

    const inp_img = getImgDataFromImgElem(input_elem);
    imagedata2Canvas(inp_img.c, output_elem, inp_img.w, inp_img.h);
    

    const shaderModuleInterpolate = device.createShaderModule({
        code: await read_shader( 'shaders_f32/interpolate.wgsl')
    });
    const shaderModuleAddition = device.createShaderModule({
        code: await read_shader( 'shaders_f32/addition.wgsl')
    });
    const shaderModuleConvRrdb = device.createShaderModule({
        code: await read_shader( 'shaders_f32/conv2d_allch_rrdb.wgsl')
    });
    const shaderModuleConvRrdbTwoBuff = device.createShaderModule({
        code: await read_shader( 'shaders_f32/conv2d_allch_rrdb_twobuff.wgsl')
    });
    const shaderModuleConvRrdbLReLU = device.createShaderModule({
        code: await read_shader( 'shaders_f32/conv2d_allch_rrdb_lrelu.wgsl')
    });
    const shaderModuleConvRrdbTwoBuffLReLU = device.createShaderModule({
        code: await read_shader( 'shaders_f32/conv2d_allch_rrdb_twobuff_lrelu.wgsl')
    });
    const shaderModuleReLURrdb = device.createShaderModule({
        code: await read_shader( 'shaders_f32/leakyrelu_rrdb.wgsl')
    });
    const shaderModuleScaleResRrdb = device.createShaderModule({
        code: await read_shader( 'shaders_f32/scaleandresidual_rrdb.wgsl')
    });
    const shaderModuleScaleResRrdbInplace = device.createShaderModule({
        code: await read_shader( 'shaders_f32/scaleandresidual_rrdb_inplace.wgsl')
    });
    // const shaderModuleScaleResRrdbRev = device.createShaderModule({
    //     code: await read_shader( 'shaders_f32/scaleandresidual_rrdb_rev.wgsl')
    // });
    // const shaderModuleCopy = device.createShaderModule({
    //     code: await read_shader( 'shaders_f32/copy.wgsl')
    // });


    function copy_mat_gpu(floatArr, arrclass, usage){
        // get GPU pointer for CPU float32 array and copy data to it
        // Need to change this if we are doing quantization
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

    

    function copy_mat_inpimg(img_c_in){
        // for one inner rrdb, we will make two buffers to swap between
        // for the first we must copy the first rdb_in data to it
        const gpuArrayRead = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * (3) * inp_img.w * inp_img.h,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const gpuArrayRNGrdb = gpuArrayRead.getMappedRange(0, Float32Array.BYTES_PER_ELEMENT * 3 * inp_img.w * inp_img.h);
        new Float32Array(gpuArrayRNGrdb).set(img_c_in);
        gpuArrayRead.unmap();
        return gpuArrayRead;

    }

    function get_mat_empty(ch_size, width, height){
        const w = width === undefined ? inp_img.w : width;
        const h = height === undefined ? inp_img.h : height;
        // for one inner rrdb, we will make two buffers to swap between
        // for the first we must copy the first rdb_in data to it
        const gpuArrayRead = device.createBuffer({
            mappedAtCreation: false,
            size: Float32Array.BYTES_PER_ELEMENT * (ch_size) * w * h,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        return gpuArrayRead;
    }


    function device2device(inputBuffer, inputOffset, outputBuffer, outputOffset, length){
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            inputBuffer /* source buffer */,
            inputOffset * Float32Array.BYTES_PER_ELEMENT /* source offset */,
            outputBuffer /* destination buffer */,
            outputOffset * Float32Array.BYTES_PER_ELEMENT /* destination offset */,
            length * Float32Array.BYTES_PER_ELEMENT /* size */
        );

        // Submit copy commands.
        const copyCommands = copyEncoder.finish();
        device.queue.submit([copyCommands]);
    }

    function preloadWeightsAndBias(_modeldata){
        // allocate one large gpu array for weights and one for bias
        const offsetsw = {};
        const sizesw = {};
        let offsetw = 0;
        const offsetsb = {};
        const sizesb = {};
        let offsetb = 0;
        for(let layername in _modeldata){
            offsetsw[layername] = offsetw;
            sizesw[layername] = _modeldata[layername].w.length
            offsetw += _modeldata[layername].w.length;
            offsetsb[layername] = offsetb;
            sizesb[layername] = _modeldata[layername].b.length
            offsetb += _modeldata[layername].b.length;
        }
        const gpuArrayWeight = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * offsetw,
            usage: GPUBufferUsage.STORAGE
        });
        const gpuArrayBias = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * offsetb,
            usage: GPUBufferUsage.STORAGE
        });

        for(let layername in _modeldata){
            const gpuArrayRNGW = gpuArrayWeight.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsw[layername],
                Float32Array.BYTES_PER_ELEMENT * sizesw[layername]);
            new Float32Array(gpuArrayRNGW).set(_modeldata[layername].w);
            const gpuArrayRNGB = gpuArrayBias.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsb[layername],
                Float32Array.BYTES_PER_ELEMENT * sizesb[layername]);
            new Float32Array(gpuArrayRNGB).set(_modeldata[layername].b);
        }
        gpuArrayWeight.unmap();
        gpuArrayBias.unmap();

        return {
            offsetsw: offsetsw,
            offsetsb: offsetsb,
            wbuf: gpuArrayWeight,
            bbuf: gpuArrayBias
        }



        //Float32Array, GPUBufferUsage.STORAGE
    }


    async function gpuexec_conv_rrdb(buffer, inputshape, weight, weightshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModule) {

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
                        type: "storage"
                    }
                },
                {
                    binding: 3,
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

        let aux_arr = [inputshape[1], inputshape[2]].concat(inputshape).concat(weightshape).concat([offsetw, offsetb, inputOffset, outputOffset])
        const channelIdxs = new Int32Array(aux_arr);

        const unifbuffer = copy_mat_gpu(channelIdxs, Int32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST


        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: weight
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: bias
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: buffer
                    }
                },
                {
                    binding: 3,
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
        const x = Math.ceil(inputshape[1] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(inputshape[2] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(weightshape[0] / 4); 
        passEncoder.dispatch(x, y, z);

        // passEncoder.dispatch(inputshape[1]/4, inputshape[2]/4, weightshape[0]/4);
        //passEncoder.dispatch(1, 1);
        //passEncoder.dispatch(1, 1, 1);*
        passEncoder.endPass();

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        unifbuffer.destroy();



    }

    async function gpuexec_conv_rrdb_twobuff(inputBuff, outputBuff, inputshape, weight, weightshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModule) {

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
                        type: "storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },{
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

        let aux_arr = [inputshape[1], inputshape[2]].concat(inputshape).concat(weightshape).concat([offsetw, offsetb, inputOffset, outputOffset])
        const channelIdxs = new Int32Array(aux_arr);

        const unifbuffer = copy_mat_gpu(channelIdxs, Int32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST


        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: weight
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: bias
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: outputBuff
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: unifbuffer
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: inputBuff
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
        const x = Math.ceil(inputshape[1] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(inputshape[2] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(weightshape[0] / 4); 
        passEncoder.dispatch(x, y, z);
        //passEncoder.dispatch(1, 1);
        //passEncoder.dispatch(1, 1, 1);*
        passEncoder.endPass();

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        unifbuffer.destroy();



    }

    async function gpuexec_relu_rrdb(writebuf, inputSize, outputOffset, shaderModule) {
        const computePipeline = device.createComputePipeline({
            compute: {
                module: shaderModule,
                entryPoint: "main"
            }
        });

        // Bind group
        const matrixSize = new Uint32Array(inputSize.concat([outputOffset]));

        const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: writebuf
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
        const x = Math.ceil(inputSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(inputSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(inputSize[0] / 4);
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        unifbuffer.destroy();
    }


    async function gpuexec_readbuf(output, outputsize, offset){
        const commandEncoder = device.createCommandEncoder();

        if(offset === undefined){
            offset = 0;
        }

        // Get a GPU buffer for reading in an unmapped state.
        const gpuReadBuffer = device.createBuffer({
            size: outputsize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Encode commands for copying buffer to buffer.
        commandEncoder.copyBufferToBuffer(
            output /* source buffer */,
            offset * Float32Array.BYTES_PER_ELEMENT /* source offset */,
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

    async function scale_residual_rrdb(inputBuff, outputBuff, inputOffset, outputOffset, in_ch_count, reverse) {
        const inputSize = [in_ch_count, inp_img.h, inp_img.w];




        // Bind group
        const matrixSize = new Uint32Array(inputSize.concat([inputOffset, outputOffset]));

        const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST

        let entries;
        let module;
        if(inputBuff === outputBuff){
            module = shaderModuleScaleResRrdbInplace;
            entries = [
                {
                    binding: 0,
                    resource: {
                        buffer: inputBuff
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: unifbuffer
                    }
                }
            ]
        }else{
            module = shaderModuleScaleResRrdb;
            entries = [
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
        }

        const computePipeline = device.createComputePipeline({
            compute: {
                module: module,
                entryPoint: "main"
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0 /* index */),
            entries: entries
        });
        const commandEncoder = device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const x = Math.ceil(inputSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
        const y = Math.ceil(inputSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
        const z = Math.ceil(inputSize[0] / 4);
        passEncoder.dispatch(x, y, z);
        passEncoder.endPass();
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);
        unifbuffer.destroy();

    }

    async function matrix_addition(inputBuff, outputBuff, in_ch_count) {
        const inputSize = [in_ch_count, inp_img.h, inp_img.w];

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
        unifbuffer.destroy();
    }

    async function up_resolution_dyn(inputBuff, outputBuff, inputSize) {
        //inputSize = [input.length / (input_shape[0] * input_shape[1]), input_shape[0], input_shape[1]];
        //const inputSize = [in_ch_count, inp_img.h, inp_img.w];
        // const outputBuff =  get_mat_gpu_output(4 * inputSize[0] * input_shape[0] * input_shape[1]);
        // const inputBuff = copy_mat_gpu(input, Float32Array, GPUBufferUsage.STORAGE);
        // const outputsize = Float32Array.BYTES_PER_ELEMENT * ((4 * inputSize[0] * input_shape[0] * input_shape[1]));


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
        //const res = await gpuexec_readbuf(outputBuff, outputsize);
        // outputBuff.destroy();
        // inputBuff.destroy();
        unifbuffer.destroy();
        //return res;
    }

    async function conv_fwd_rrdb(inputBuff, outputBuff, in_ch_count, inp_shape, weight, wshape, offsetw, bias, bshape, offsetb, inputOffset, outputOffset, relu, dbg){
        
        if(inputBuff === outputBuff){
            if (relu) {
                await gpuexec_conv_rrdb(outputBuff, [in_ch_count, inp_shape[1], inp_shape[0]],
                    weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdbLReLU);
            } else {
                await gpuexec_conv_rrdb(outputBuff, [in_ch_count, inp_shape[1], inp_shape[0]],
                    weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdb);
            }            
        }else{
            if (relu) {
                await gpuexec_conv_rrdb_twobuff(inputBuff, outputBuff, [in_ch_count, inp_shape[1], inp_shape[0]],
                    weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdbTwoBuffLReLU);
            } else {
                await gpuexec_conv_rrdb_twobuff(inputBuff, outputBuff, [in_ch_count, inp_shape[1], inp_shape[0]],
                    weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdbTwoBuff);
            }
        }
        // if (relu) {
        //     await gpuexec_relu_rrdb(outputBuff, [wshape[0], inp_shape[0], inp_shape[1]], outputOffset, shaderModuleReLURrdb)
        // }
    }

    async function esrgan(){


        const layerdatabufs = await preloadWeightsAndBias(_modeldata);

        async function eval_conv_rrdb(name, inputBuff, outputBuff, in_ch_count, inputOffset, outputOffset, inp_shape, relu, dbg){
            //console.log(_modeldata[name])
            console.log(name);
            return await conv_fwd_rrdb(
                inputBuff, outputBuff,
                in_ch_count, inp_shape,
                layerdatabufs.wbuf,
                _modeldata[name].wshape,
                layerdatabufs.offsetsw[name],
                layerdatabufs.bbuf,
                _modeldata[name].bshape,
                layerdatabufs.offsetsb[name],
                inputOffset, outputOffset,
                relu, dbg
            );
        }

        const feaBuf = get_mat_empty(64);
        const inpImgBuf = copy_mat_inpimg(inp_img.c);
        await eval_conv_rrdb('conv_first', inpImgBuf, feaBuf, 3, 0, 0, [inp_img.h, inp_img.w], false);
        inpImgBuf.destroy();

        const rbd_swapbuf = get_mat_empty(64 + 192);
        const rrbd_swapbuf = get_mat_empty(64);
        device2device(feaBuf, 0, rbd_swapbuf, 0, (64) * inp_img.w * inp_img.h);
        device2device(feaBuf, 0, rrbd_swapbuf, 0, (64) * inp_img.w * inp_img.h);
        //console.log(fea)
        for (let rrdb_chunk = 0; rrdb_chunk < 23; rrdb_chunk++) {
            //let rdb_in = rrdb_in;

            for (let rdb = 1; rdb <= 3; rdb ++) {


                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv1', rbd_swapbuf, rbd_swapbuf,
                    64, 0, (64) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true );



                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv2', rbd_swapbuf, rbd_swapbuf,
                    96, 0, (96) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv3', rbd_swapbuf, rbd_swapbuf,
                    128, 0, (128) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv4', rbd_swapbuf, rbd_swapbuf,
                    160, 0, (160) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv5', rbd_swapbuf, rbd_swapbuf,
                    192, 0, (192) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], false);


                await scale_residual_rrdb(rbd_swapbuf, rbd_swapbuf, (192) * inp_img.w * inp_img.h, 0, 64);

            }

            await scale_residual_rrdb(rbd_swapbuf, rrbd_swapbuf, 0,(0) * inp_img.w * inp_img.h, 64);
            if(rrdb_chunk != 22){
                device2device(rrbd_swapbuf, 0, rbd_swapbuf, 0, (64) * inp_img.w * inp_img.h);
            }

        }
        //rdb_swafbuf.destroy()
        //rrdb_swafbuf.destroy()
        //rrdb_in = await gpuexec_readbuf(rrbd_swapbuf, Float32Array.BYTES_PER_ELEMENT * (64) * inp_img.w * inp_img.h,  (0) * inp_img.w * inp_img.h);
        await eval_conv_rrdb('trunk_conv', rrbd_swapbuf, rbd_swapbuf, 64, 0, 0, [inp_img.h, inp_img.w], false);
        rrbd_swapbuf.destroy()
        await matrix_addition(rbd_swapbuf, feaBuf, 64);
        rbd_swapbuf.destroy()

        // console.log(fea)
        const upres1_swapbuf1 = get_mat_empty(64, inp_img.w * 2, inp_img.h * 2);
        await up_resolution_dyn(feaBuf, upres1_swapbuf1,  [64, inp_img.h, inp_img.w]);
        feaBuf.destroy();

        const upres1_swapbuf2 = get_mat_empty(64, inp_img.w * 2, inp_img.h * 2);

        await eval_conv_rrdb('upconv1', upres1_swapbuf1, upres1_swapbuf2, 64, 0, 0, [2 * inp_img.h, 2 * inp_img.w], true);
        upres1_swapbuf1.destroy();

        const upres2_swapbuf1 = get_mat_empty(64, inp_img.w * 4, inp_img.h * 4);
        await up_resolution_dyn(upres1_swapbuf2, upres2_swapbuf1,  [64, 2 * inp_img.h, 2 * inp_img.w]);
        upres1_swapbuf2.destroy();

        const upres2_swapbuf2 = get_mat_empty(64, inp_img.w * 4, inp_img.h * 4);
        await eval_conv_rrdb('upconv2', upres2_swapbuf1, upres2_swapbuf2, 64, 0, 0, [4 * inp_img.h, 4 * inp_img.w], true);
        await eval_conv_rrdb('HRconv', upres2_swapbuf2, upres2_swapbuf1, 64, 0, 0, [4 * inp_img.h, 4 * inp_img.w], true);
        upres2_swapbuf2.destroy();

        const outImgBuf = get_mat_empty(3, 4 * inp_img.w, 4 * inp_img.h);
        eval_conv_rrdb('conv_last', upres2_swapbuf1, outImgBuf, 64, 0, 0, [4 * inp_img.h, 4 * inp_img.w], false);
        upres2_swapbuf1.destroy();

        let outImg = await gpuexec_readbuf(outImgBuf, Float32Array.BYTES_PER_ELEMENT * (3) * inp_img.w * 4 * inp_img.h * 4,  (0) * inp_img.w * inp_img.h);
        outImgBuf.destroy();

        // delete weights and biases from gpu
        layerdatabufs.wbuf.destroy();
        layerdatabufs.bbuf.destroy();

        return outImg;

    }
    var startTime = performance.now()

    const nn_result = await esrgan();
    console.log(nn_result)

    var endTime = performance.now()

    console.log(`NN run took ${endTime - startTime} milliseconds`)

    imagedata2Canvas(nn_result, output_elem, inp_img.w*4, inp_img.h*4);
    document.getElementById('imageUpload').disabled = false;

  }
  