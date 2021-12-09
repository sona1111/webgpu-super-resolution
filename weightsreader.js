
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

async function storeModelData(store_key, url){
    if(store_key in modeldata){
        return;
    }
    document.getElementById('imageUpload').disabled = true;
    document.getElementById('dataload_status').textContent = `Downloading ${store_key}...`;
    let meta = await makeRequest("GET", `${url}/modelinfo.json`, 'json');
    const arrayclass = meta.is_quantized === true ? Int8Array : Float32Array;
    const progress = document.getElementById('download_progress');
    progress.value = 0;
    // by default, use local storage for cache
    const result = {};
    const num_layers  = Object.keys(meta.layers).length;
    let progress_done = 0;
    progress.max = num_layers;

    for(let layer of meta.layers){

        const alreadyExist = await getLDBAsync(`${url}/${layer.name}`);
        if(alreadyExist){
            //console.log(`using cached ${url}/${layer.name}`)
            result[layer.name] = JSON.parse(alreadyExist);
            result[layer.name].w = await getLDBAsync(`${url}/${layer.name}/w`);
            result[layer.name].b = await getLDBAsync(`${url}/${layer.name}/b`);
        }else{
            const layerWeights = await downloadLayerWeights(url, layer, arrayclass);
            result[layer.name] = layerWeights;
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
        progress_done += 1;
        progress.value = progress_done;
    }

    modeldata[store_key] = result;
    document.getElementById('dataload_status').textContent = `Ready!`;
    document.getElementById('imageUpload').disabled = false;
}

function quantizeToInt8(float32){
    // likely this doesn't actually work lol
    return new Int8Array(float32.map(x => x * 127.5-0.5));
}


