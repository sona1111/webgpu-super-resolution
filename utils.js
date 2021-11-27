function _getImgDataFromImgElem(imgElem){
    const canvas = document.createElement('canvas');
    canvas.width = imgElem.width;
    canvas.height = imgElem.height;
    canvas.getContext('2d').drawImage(imgElem, 0, 0, imgElem.width, imgElem.height);
    const pixelData = canvas.getContext('2d').getImageData(0, 0, imgElem.width, imgElem.height).data;
    const r = new Float32Array(pixelData.length/4);
    const g = new Float32Array(pixelData.length/4);
    const b = new Float32Array(pixelData.length/4);
    let j=0;
    for(let i=0;i<pixelData.length;i+=4){
        r[j]=pixelData[i];
        g[j]=pixelData[i+1];
        b[j]=pixelData[i+2];
        j++;
    }
    return {
        w: imgElem.width,
        h: imgElem.height,
        r: r,
        g: g,
        b: b
    };
}

function getImgDataFromImgElem(imgElem){
    const canvas = document.createElement('canvas');
    canvas.width = imgElem.width;
    canvas.height = imgElem.height;
    canvas.getContext('2d').drawImage(imgElem, 0, 0, imgElem.width, imgElem.height);
    const pixelData = canvas.getContext('2d').getImageData(0, 0, imgElem.width, imgElem.height).data;
    const color = new Float32Array(pixelData.length*3/4);
    const color_ch_len = pixelData.length/4;

    for(let i=0;i<pixelData.length;i+=4){
        color[i/4]=pixelData[i]; // red
        color[(i/4) + color_ch_len]=pixelData[i+1];
        color[(i/4) + (color_ch_len*2)]=pixelData[i+2];

    }
    return {
        w: imgElem.width,
        h: imgElem.height,
        c: color
    };
}