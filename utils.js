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
        color[i/4]=pixelData[i]/255.0; // red
        color[(i/4) + color_ch_len]=pixelData[i+1]/255.0;
        color[(i/4) + (color_ch_len*2)]=pixelData[i+2]/255.0;

    }
    return {
        w: imgElem.width,
        h: imgElem.height,
        c: color
    };
}

function imagedata2Canvas(array, elem, width, height){
    const imgDataArr = new Uint8ClampedArray(array.length * (4/3));
    const color_ch_len = array.length/3;
    for(let i=0; i<imgDataArr.length;i+=4){
        imgDataArr[i] = array[i/4]*255;
        imgDataArr[i+1] = array[(i/4) + color_ch_len]*255;
        imgDataArr[i+2] = array[(i/4) + (color_ch_len*2)]*255;
        imgDataArr[i+3] = 255;
    }

    const imgData = new ImageData(imgDataArr, width);
    elem.width = width;
    elem.height = height;
    const ctx = elem.getContext('2d');
    ctx.putImageData(imgData, 0, 0);



}