
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

async function getModelWeights(url){

    let result = await makeRequest("GET", `${url}/modelinfo.json`, 'json');

    const layerdata = {}

    for (let layer of result.layers){
        // note, theoretically with this model we would start to evaluate the earlier layers before
        // the later ones are downloaded
        let weights = new Float32Array(await makeRequest("GET", `${url}/${layer}.weight.bin`, 'arraybuffer'));
        let bias = new Float32Array(await makeRequest("GET", `${url}/${layer}.bias.bin`, 'arraybuffer'));
        layerdata[layer] = {'w': weights, 'b': bias};

    }

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

getModelWeights("ESRGAN/small");