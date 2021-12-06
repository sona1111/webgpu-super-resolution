

function initmain(){



    document.getElementById('imageUpload').addEventListener('change', async function() {
        if (this.files && this.files[0]) {
            var img = document.querySelector('img');
            img.onload = async function (){
                URL.revokeObjectURL(img.src);  // no longer needed, free memory
                const output_elem = document.getElementById('result')
                document.getElementById('networkrunningloader').style.display = 'inline-block';
                await run_nn(img, output_elem);
                document.getElementById('networkrunningloader').style.display = 'none';
            }

            img.src = URL.createObjectURL(this.files[0]); // set src to blob url
        }
    });

    document.getElementById('demo1').addEventListener('click', async function() {
        var img = document.getElementById('sm_img');
        const output_elem = document.getElementById('result')
        document.getElementById('networkrunningloader').style.display = 'inline-block';
        await run_nn(img, output_elem);
        document.getElementById('networkrunningloader').style.display = 'none';

    });

    document.getElementById('demo2').addEventListener('click', async function() {
        var img = document.getElementById('med_img');
        const output_elem = document.getElementById('result')
        document.getElementById('networkrunningloader').style.display = 'inline-block';
        await run_nn(img, output_elem);
        document.getElementById('networkrunningloader').style.display = 'none';
    });

    document.getElementById('demo3').addEventListener('click', async function() {
        var img = document.getElementById('lrg_img');
        const output_elem = document.getElementById('result')
        document.getElementById('networkrunningloader').style.display = 'inline-block';
        await run_nn(img, output_elem);
        document.getElementById('networkrunningloader').style.display = 'none';
    });



}

