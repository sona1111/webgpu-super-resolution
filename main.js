

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
            img.onload = async function (){
                URL.revokeObjectURL(img.src);  // no longer needed, free memory

                //document.getElementById('networkrunningloader').style.display = 'inline-block';
                document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
                await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
                document.getElementById('resultpanel').style.backgroundColor = '';
                //document.getElementById('networkrunningloader').style.display = 'none';
            }

            img.src = URL.createObjectURL(this.files[0]); // set src to blob url
        }
    });
    document.getElementById('mannualrerun').addEventListener('click', async function() {
        document.getElementById('imageUpload').dispatchEvent(new Event('change'));
    });

    document.getElementById('demo1').addEventListener('click', async function() {
        var img = document.getElementById('sm_img');

        //document.getElementById('networkrunningloader').style.display = 'inline-block';
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        //document.getElementById('networkrunningloader').style.display = 'none';

    });

    document.getElementById('demo2').addEventListener('click', async function() {
        var img = document.getElementById('med_img');

        //document.getElementById('networkrunningloader').style.display = 'inline-block';
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        //document.getElementById('networkrunningloader').style.display = 'none';
    });

    document.getElementById('demo3').addEventListener('click', async function() {
        var img = document.getElementById('lrg_img');

        //document.getElementById('networkrunningloader').style.display = 'inline-block';
        document.getElementById('resultpanel').style.backgroundColor = '#d1806b';
        await run_nn(img, output_elem, status_elem, gpumem_elem, progress_elem, parseInt(num_dense_blocks_elem.value));
        document.getElementById('resultpanel').style.backgroundColor = '';
        //document.getElementById('networkrunningloader').style.display = 'none';
    });



}

