



storeModelData("RRDB_ESRGAN_x4", "ESRGAN_py/RRDB_ESRGAN_x4");

document.getElementById('imageUpload').addEventListener('change', function() {
    if (this.files && this.files[0]) {
        var img = document.querySelector('img');
        img.onload = () => {
            URL.revokeObjectURL(img.src);  // no longer needed, free memory
            const output_elem = document.getElementById('result')
            run_nn(img, output_elem, modeldata["RRDB_ESRGAN_x4"]);

        }

        img.src = URL.createObjectURL(this.files[0]); // set src to blob url
    }
});

document.getElementById('demo1').addEventListener('click', function() {
    var img = document.getElementById('sm_img');
    const output_elem = document.getElementById('result')
    run_nn(img, output_elem, modeldata["RRDB_ESRGAN_x4"]);
});

document.getElementById('demo2').addEventListener('click', function() {
    var img = document.getElementById('med_img');
    const output_elem = document.getElementById('result')
    run_nn(img, output_elem, modeldata["RRDB_ESRGAN_x4"]);
});

document.getElementById('demo3').addEventListener('click', function() {
    var img = document.getElementById('lrg_img');
    const output_elem = document.getElementById('result')
    run_nn(img, output_elem, modeldata["RRDB_ESRGAN_x4"]);
});


