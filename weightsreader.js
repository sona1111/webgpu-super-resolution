
function makeRequest(method, url, responsetype) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.open(method, url);
        xhr.responseType = responsetype;
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send();
    });
}

async function downloadLayerWeights(url, layer, arrayclass){


    // note, theoretically with this model we would start to evaluate the earlier layers before
    // the later ones are downloaded
    let weights = new arrayclass(await makeRequest("GET", `${url}/${layer.name}.weight.bin`, 'arraybuffer'));
    let bias = new arrayclass(await makeRequest("GET", `${url}/${layer.name}.bias.bin`, 'arraybuffer'));


    return {'w': weights, 'b': bias, 'wshape':layer.wshape, 'bshape':layer.bshape};

}

function getLDBAsync(key) {
    return new Promise(function(resolve, reject) {
        ldb.get(key, function(data){
            resolve(data);
        });
    });
}

async function read_shader_and_create(path){
    const conv2dc = await fetch(path);
    const code = await conv2dc.text();
    const shadermodule = current_device.createShaderModule({
        code: code
    });
    return shadermodule;
}


async function storeModelData(store_key, url){
    // if(store_key in modeldata){
    //     return;
    // }
    if(device_model_pointers !== null){
        device_model_pointers.wbuf.destroy();
        device_model_pointers.bbuf.destroy();
    }

    document.getElementById('imageUpload').disabled = true;
    document.getElementById('dataload_status').textContent = `Downloading ${store_key}...`;
    let meta = await makeRequest("GET", `${url}/modelinfo.json`, 'json');
    device_model_meta = meta.layers;
    const arrayclass = meta.is_quantized === true ? Int8Array : Float32Array;
    const progress = document.getElementById('download_progress');
    progress.value = 0;
    // by default, use local storage for cache
    const result = {};
    const num_layers  = Object.keys(meta.layers).length;
    let progress_done = 0;
    progress.max = num_layers;

    // calculate buffer sizes needed

    const offsetsw = {};
    const sizesw = {};
    let offsetw = 0;
    const offsetsb = {};
    const sizesb = {};
    let offsetb = 0;

    for(let layer of meta.layers){
        offsetsw[layer.name] = offsetw;
        sizesw[layer.name] = layer.wshape.reduce((a, b)=> a*b, 1);
        offsetw += layer.wshape.reduce((a, b)=> a*b, 1);
        offsetsb[layer.name] = offsetb;
        sizesb[layer.name] = layer.bshape.reduce((a, b)=> a*b, 1);
        offsetb += layer.bshape.reduce((a, b)=> a*b, 1);
        device_model_meta[layer.name] = {
            wshape: layer.wshape,
            bshape: layer.bshape
        }
    }

    const device = await getDevice();
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

    for(let layer of meta.layers){

        const alreadyExist = await getLDBAsync(`${url}/${layer.name}`);
        let cpuWeight = null;
        let cpuBias = null;
        if(alreadyExist){
            //console.log(`using cached ${url}/${layer.name}`)
            //result[layer.name] = JSON.parse(alreadyExist);
            // result[layer.name].w = await getLDBAsync(`${url}/${layer.name}/w`);
            // result[layer.name].b = await getLDBAsync(`${url}/${layer.name}/b`);
            cpuWeight = await getLDBAsync(`${url}/${layer.name}/w`);
            cpuBias = await getLDBAsync(`${url}/${layer.name}/b`);
        }else{
            const layerWeights = await downloadLayerWeights(url, layer, arrayclass);
            //result[layer.name] = layerWeights;
            cpuWeight = layerWeights.w;
            cpuBias = layerWeights.b;
            try{
                ldb.set(`${url}/${layer.name}`, JSON.stringify({
                    'wshape': layerWeights.wshape,
                    'bshape': layerWeights.bshape
                }));
                ldb.set(`${url}/${layer.name}/w`, layerWeights.w)
                ldb.set(`${url}/${layer.name}/b`, layerWeights.b)
                console.log(`Stored ${url} to localstorage`);
            }catch (error){
                console.log(`UNABLE to store ${url} to localstorage`);
            }
        }

        // upload to gpu
        const gpuArrayRNGW = gpuArrayWeight.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsw[layer.name],
            Float32Array.BYTES_PER_ELEMENT * sizesw[layer.name]);
        new Float32Array(gpuArrayRNGW).set(cpuWeight);
        const gpuArrayRNGB = gpuArrayBias.getMappedRange(Float32Array.BYTES_PER_ELEMENT * offsetsb[layer.name],
            Float32Array.BYTES_PER_ELEMENT * sizesb[layer.name]);
        new Float32Array(gpuArrayRNGB).set(cpuBias);


        progress_done += 1;
        progress.value = progress_done;
    }

    gpuArrayWeight.unmap();
    gpuArrayBias.unmap();

    //modeldata[store_key] = result;
    document.getElementById('dataload_status').textContent = `Ready!`;
    document.getElementById('imageUpload').disabled = false;

    device_model_pointers = {
        offsetsw: offsetsw,
        offsetsb: offsetsb,
        wbuf: gpuArrayWeight,
        bbuf: gpuArrayBias
    }

    if(shader_modules === null){
        shader_modules = {
            shaderModuleInterpolate: 'shaders_f32/interpolate.wgsl',
            shaderModuleAddition:  'shaders_f32/addition.wgsl',
            shaderModuleConvRrdb: 'shaders_f32/conv2d_allch_rrdb_unrolled.wgsl',
            shaderModuleConvRrdbTwoBuff: 'shaders_f32/conv2d_allch_rrdb_twobuff_unrolled.wgsl',
            shaderModuleConvRrdbLReLU: 'shaders_f32/conv2d_allch_rrdb_lrelu_unrolled_v2.wgsl',
            shaderModuleConvRrdbTwoBuffLReLU: 'shaders_f32/conv2d_allch_rrdb_twobuff_lrelu_unrolled.wgsl',
            shaderModuleReLURrdb: 'shaders_f32/leakyrelu_rrdb.wgsl',
            shaderModuleScaleResRrdb: 'shaders_f32/scaleandresidual_rrdb.wgsl',
            shaderModuleScaleResRrdbInplace: 'shaders_f32/scaleandresidual_rrdb_inplace.wgsl'
        }
        for(let modulename in shader_modules){
            shader_modules[modulename] = read_shader_and_create(shader_modules[modulename]);
        }
        for(let modulename in shader_modules){
            shader_modules[modulename] = await shader_modules[modulename];
        }
    }



}

function quantizeToInt8(float32){
    // likely this doesn't actually work lol
    return new Int8Array(float32.map(x => x * 127.5-0.5));
}


