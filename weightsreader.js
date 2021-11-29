
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

async function downloadLayerWeights(url, layer){


    // note, theoretically with this model we would start to evaluate the earlier layers before
    // the later ones are downloaded
    let weights = new Float32Array(await makeRequest("GET", `${url}/${layer.name}.weight.bin`, 'arraybuffer'));
    let bias = new Float32Array(await makeRequest("GET", `${url}/${layer.name}.bias.bin`, 'arraybuffer'));


    return {'w': weights, 'b': bias, 'wshape':layer.wshape, 'bshape':layer.bshape};

    // var oReq = new XMLHttpRequest();
    // oReq.open("GET", url, true);
    // oReq.responseType = "arraybuffer";
    //
    // const model_weights = {};
    //
    // oReq.onload = function (oEvent) {
    //     var arrayBuffer = oReq.response; // Note: not oReq.responseText
    //     if (arrayBuffer) {
    //
    //         // const parts = arrayBuffer.
    //         // var byteArray = new Uint8Array(arrayBuffer);
    //         // for (var i = 0; i < byteArray.byteLength; i++) {
    //         //     // do something with each byte in the array
    //         // }
    //     }
    // };
    //
    // oReq.send(null);
}

function getLDBAsync(key) {
    return new Promise(function(resolve, reject) {
        ldb.get(key, function(data){
            resolve(data);
        });
    });
}

async function getModelData(url){

    let meta = await makeRequest("GET", `${url}/modelinfo.json`, 'json');

    // by default, use local storage for cache
    const result = {};

    for(let layer of meta.layers){
        const alreadyExist = await getLDBAsync(`${url}/${layer.name}`);
        if(alreadyExist){
            console.log(`using cached ${url}/${layer.name}`)
            result[layer.name] = JSON.parse(alreadyExist);
            result[layer.name].w = new Float32Array(Object.values(result[layer.name].w));
            result[layer.name].b = new Float32Array(Object.values(result[layer.name].b));
            continue;
        }

        const layerWeights = await downloadLayerWeights(url, layer);
        result[layer.name] = layerWeights;
        try{
            ldb.set(`${url}/${layer.name}`, JSON.stringify(layerWeights));
            console.log(`Stored ${url} to localstorage`);
        }catch (error){
            console.log(`UNABLE to store ${url} to localstorage`);
        }
    }

    return result;
}

