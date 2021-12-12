

function initmain(){

    const output_elem = document.getElementById('result');
    const status_elem = document.getElementById('dataload_status');
    const gpumem_elem = document.getElementById('memusage');
    const progress_elem = document.getElementById('download_progress');
    const num_dense_blocks_elem = document.getElementById('numDenseBlocks');



    document.getElementById('imageUpload').addEventListener('change', async function() {
        if (this.files && this.files[0]) {
            var img = document.querySelector('img');
            img.style.display = 'none';
            console.log(this.files[0]);
            const parts = this.files[0].name.split('.');
            parts.pop();
            const filename = parts.join('.');
            img.onload = async function (){
                URL.revokeObjectURL(img.src);  // no longer needed, free memory

                document.getElementById('resultInfo').style.display = 'none';
                document.getElementById('result').setAttribute('data-imagename', filename);
                document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
                await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
                document.getElementById('resultpanel').style.backgroundColor = '';
                document.getElementById('resultInfo').style.display = 'block';
            }

            img.src = URL.createObjectURL(this.files[0]); // set src to blob url
        }
    });
    document.getElementById('mannualrerun').addEventListener('click', async function() {
        document.getElementById('imageUpload').dispatchEvent(new Event('change'));
    });

    document.getElementById('demo1').addEventListener('click', async function() {
        var img = document.getElementById('sm_img');

        document.getElementById('resultInfo').style.display = 'none';
        document.getElementById('result').setAttribute('data-imagename', 'demo1');
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        document.getElementById('resultInfo').style.display = 'block';

    });

    document.getElementById('demo2').addEventListener('click', async function() {
        var img = document.getElementById('med_img');

        document.getElementById('resultInfo').style.display = 'none';
        document.getElementById('result').setAttribute('data-imagename', 'demo2');
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        document.getElementById('resultInfo').style.display = 'block';
    });

    document.getElementById('demo3').addEventListener('click', async function() {
        var img = document.getElementById('lrg_img');

        document.getElementById('resultInfo').style.display = 'none';
        document.getElementById('result').setAttribute('data-imagename', 'demo3');
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        document.getElementById('resultInfo').style.display = 'block';
    });

    document.getElementById('downloadResultBtn').addEventListener('click', function(){
        downloadCanvas(document.getElementById('result'),
            `${document.getElementById('result').getAttribute('data-imagename')}-4x-${current_network}`);
    });



}

