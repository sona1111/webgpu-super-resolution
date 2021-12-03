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
    const shaderModuleConvRrdb = device.createShaderModule({
        code: await read_shader( 'conv2d_allch_rrdb.wgsl')
    });
    const shaderModuleConvRrdbDbg = device.createShaderModule({
        code: await read_shader( 'conv2d_allch_rrdb_dbg.wgsl')
    });
    const shaderModuleReLURrdb = device.createShaderModule({
        code: await read_shader( 'leakyrelu_rrdb.wgsl')
    });
    const shaderModuleScaleResRrdb = device.createShaderModule({
        code: await read_shader( 'scaleandresidual_rrdb.wgsl')
    });
    const shaderModuleScaleResRrdbInplace = device.createShaderModule({
        code: await read_shader( 'scaleandresidual_rrdb_inplace.wgsl')
    });
    // const shaderModuleScaleResRrdbRev = device.createShaderModule({
    //     code: await read_shader( 'scaleandresidual_rrdb_rev.wgsl')
    // });
    // const shaderModuleCopy = device.createShaderModule({
    //     code: await read_shader( 'copy.wgsl')
    // });


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

    function copy_mat_rdb(rdb_in){
        // for one inner rrdb, we will make two buffers to swap between
        // for the first we must copy the first rdb_in data to it
        const gpuArrayRead = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * (64 + 192) * inp_img.w * inp_img.h,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const gpuArrayRNGrdb = gpuArrayRead.getMappedRange(0, Float32Array.BYTES_PER_ELEMENT * 64 * inp_img.w * inp_img.h);
        new Float32Array(gpuArrayRNGrdb).set(rdb_in);
        gpuArrayRead.unmap();
        return gpuArrayRead;

    }

    function copy_mat_rrdb(rdb_in){
        // for one inner rrdb, we will make two buffers to swap between
        // for the first we must copy the first rdb_in data to it
        const gpuArrayRead = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * (64) * inp_img.w * inp_img.h,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const gpuArrayRNGrdb = gpuArrayRead.getMappedRange(0, Float32Array.BYTES_PER_ELEMENT * 64 * inp_img.w * inp_img.h);
        new Float32Array(gpuArrayRNGrdb).set(rdb_in);
        gpuArrayRead.unmap();
        return gpuArrayRead;

    }

    // function device2deviceSameBuf(buffer, inputOffset, outputOffset, length){
    //     const inputSize = [in_ch_count, inp_img.h, inp_img.w];
    //
    //
    //     const computePipeline = device.createComputePipeline({
    //         compute: {
    //             module: shaderModuleCopy,
    //             entryPoint: "main"
    //         }
    //     });
    //
    //     // Bind group
    //     const matrixSize = new Uint32Array(inputSize.concat([inputOffset, outputOffset]));
    //
    //     const unifbuffer = copy_mat_gpu(matrixSize, Uint32Array, GPUBufferUsage.STORAGE) //  | GPUBufferUsage.COPY_DST
    //
    //     const bindGroup = device.createBindGroup({
    //         layout: computePipeline.getBindGroupLayout(0 /* index */),
    //         entries: [
    //             {
    //                 binding: 0,
    //                 resource: {
    //                     buffer: inputBuff
    //                 }
    //             },
    //             {
    //                 binding: 1,
    //                 resource: {
    //                     buffer: unifbuffer
    //                 }
    //             }
    //         ]
    //     });
    //     const commandEncoder = device.createCommandEncoder();
    //
    //     const passEncoder = commandEncoder.beginComputePass();
    //     passEncoder.setPipeline(computePipeline);
    //     passEncoder.setBindGroup(0, bindGroup);
    //     const x = Math.ceil(inputSize[2] / 4); // X dimension of the grid of workgroups to dispatch.
    //     const y = Math.ceil(inputSize[1] / 4); // Y dimension of the grid of workgroups to dispatch.
    //     const z = Math.ceil(inputSize[0] / 4);
    //     passEncoder.dispatch(x, y, z);
    //     passEncoder.endPass();
    //     const gpuCommands = commandEncoder.finish();
    //     device.queue.submit([gpuCommands]);
    // }

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

    function preloadWeightsAndBias(modeldata){
        // allocate one large gpu array for weights and one for bias
        const offsetsw = {};
        const sizesw = {};
        let offsetw = 0;
        const offsetsb = {};
        const sizesb = {};
        let offsetb = 0;
        for(let layername in modeldata){
            offsetsw[layername] = offsetw;
            sizesw[layername] = modeldata[layername].w.length
            offsetw += modeldata[layername].w.length;
            offsetsb[layername] = offsetb;
            sizesb[layername] = modeldata[layername].b.length
            offsetb += modeldata[layername].b.length;
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

        for(let layername in modeldata){
            const gpuArrayRNGW = gpuArrayWeight.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsw[layername],
                Float32Array.BYTES_PER_ELEMENT * sizesw[layername]);
            new Float32Array(gpuArrayRNGW).set(modeldata[layername].w);
            const gpuArrayRNGB = gpuArrayBias.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsb[layername],
                Float32Array.BYTES_PER_ELEMENT * sizesb[layername]);
            new Float32Array(gpuArrayRNGB).set(modeldata[layername].b);
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



    async function gpuexec_conv(output, input, inputshape, weight, weightshape, bias, offsetw, offsetb, shaderModule) {

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

        let aux_arr = [inputshape[1], inputshape[2]].concat(inputshape).concat(weightshape).concat([offsetw, offsetb])
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



    async function conv_fwd(inp, inp_shape, weight, wshape, offsetw, bias, bshape, offsetb, relu){
        if(Array.isArray(inp)){  // not float32 array, standard JS array
            inp = vstack(inp);
        }
        const output =  get_mat_gpu_output(wshape[0] * inp_shape[0] * inp_shape[1]);
        const input = copy_mat_gpu(inp, Float32Array, GPUBufferUsage.STORAGE);
        const outputsize = Float32Array.BYTES_PER_ELEMENT * ((wshape[0] * inp_shape[0] * inp_shape[1]));

        const in_ch_count = inp.length / (inp_shape[0] * inp_shape[1]);

        await gpuexec_conv(output, input, [in_ch_count, inp_shape[1], inp_shape[0]],
            weight, wshape, bias, offsetw, offsetb, shaderModuleConv);

        if (relu) {
            await gpuexec_relu(output, [wshape[0], inp_shape[0], inp_shape[1]], shaderModuleReLU)
        }
        const res = await gpuexec_readbuf(output, outputsize);
        output.destroy();
        input.destroy();
        return res;
    }

    async function conv_fwd_rrdb(buffer, in_ch_count, inp_shape, weight, wshape, offsetw, bias, bshape, offsetb, inputOffset, outputOffset, relu, dbg){

        if (dbg){
            await gpuexec_conv_rrdb(buffer, [in_ch_count, inp_shape[1], inp_shape[0]],
                weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdbDbg);
        }else{
            await gpuexec_conv_rrdb(buffer, [in_ch_count, inp_shape[1], inp_shape[0]],
                weight, wshape, bias, offsetw, offsetb, inputOffset, outputOffset, shaderModuleConvRrdb);
        }


        if (relu) {
            await gpuexec_relu_rrdb(buffer, [wshape[0], inp_shape[0], inp_shape[1]], outputOffset, shaderModuleReLURrdb)
        }

    }

    async function esrgan(){


        const layerdatabufs = await preloadWeightsAndBias(modeldata);

        async function eval_conv(name, inp, inp_shape, relu){
            //console.log(modeldata[name])
            console.log(name);
            return await conv_fwd(
                inp,
                inp_shape,
                layerdatabufs.wbuf,
                modeldata[name].wshape,
                layerdatabufs.offsetsw[name],
                layerdatabufs.bbuf,
                modeldata[name].bshape,
                layerdatabufs.offsetsb[name],
                relu
            );
        }

        async function eval_conv_rrdb(name, buffer, in_ch_count, inputOffset, outputOffset, inp_shape, relu, dbg){
            //console.log(modeldata[name])
            console.log(name);
            return await conv_fwd_rrdb(
                buffer,
                in_ch_count, inp_shape,
                layerdatabufs.wbuf,
                modeldata[name].wshape,
                layerdatabufs.offsetsw[name],
                layerdatabufs.bbuf,
                modeldata[name].bshape,
                layerdatabufs.offsetsb[name],
                inputOffset, outputOffset,
                relu, dbg
            );
        }


        let fea = await eval_conv('conv_first', inp_img.c, [inp_img.h, inp_img.w], false);

        let rrdb_in = fea;
        const rbd_swapbuf = copy_mat_rdb(rrdb_in);
        const rrbd_swapbuf = copy_mat_rrdb(rrdb_in);
        //console.log(fea)
        for (let rrdb_chunk = 0; rrdb_chunk < 23; rrdb_chunk++) {
            //let rdb_in = rrdb_in;

            for (let rdb = 1; rdb <= 3; rdb ++) {


                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv1', rbd_swapbuf,
                    64, 0, (64) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true );



                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv2', rbd_swapbuf,
                    96, 0, (96) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv3', rbd_swapbuf,
                    128, 0, (128) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv4', rbd_swapbuf,
                    160, 0, (160) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], true);

                await eval_conv_rrdb('RRDB_trunk.' +  rrdb_chunk + '.RDB' + rdb + '.conv5', rbd_swapbuf,
                    192, 0, (192) * inp_img.w * inp_img.h,
                    [inp_img.h, inp_img.w], false);


                await scale_residual_rrdb(rbd_swapbuf, rbd_swapbuf, (192) * inp_img.w * inp_img.h, 0, 64);

            }

            await scale_residual_rrdb(rbd_swapbuf, rrbd_swapbuf, 0,(0) * inp_img.w * inp_img.h, 64);
            if(rrdb_chunk != 22){
                device2device(rrbd_swapbuf, 0, rbd_swapbuf, 0, (64) * inp_img.w * inp_img.h);
            }


            // if(rrdb_chunk == 1){
            //     const res = await gpuexec_readbuf(rrbd_swapbuf, Float32Array.BYTES_PER_ELEMENT * (64) * inp_img.w * inp_img.h,  (0) * inp_img.w * inp_img.h);
            //     return res;
            // }


            // console.log(rrdb_in)
        }
        rrdb_in = await gpuexec_readbuf(rrbd_swapbuf, Float32Array.BYTES_PER_ELEMENT * (64) * inp_img.w * inp_img.h,  (0) * inp_img.w * inp_img.h);
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

        // delete weights and biases from gpu
        layerdatabufs.wbuf.destroy();
        layerdatabufs.bbuf.destroy();

        return fea;

    }
    var startTime = performance.now()

    const nn_result = await esrgan();
    console.log(nn_result)

    var endTime = performance.now()

    console.log(`NN run took ${endTime - startTime} milliseconds`)

    imagedata2Canvas(nn_result, document.getElementById('result'), inp_img.w*4, inp_img.h*4);


  })();
  