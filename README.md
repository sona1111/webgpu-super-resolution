WebGPU Image Super Resolution
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Proejct**

* Paul (San) Jewell
  * [LinkedIn](https://www.linkedin.com/in/paul-jewell-2aba7379), [work website](
    https://www.biociphers.org/paul-jewell-lab-member), [personal website](https://gitlab.com/inklabapp), [twitter](https://twitter.com/inklabapp), etc.
* Tested on: Linux pop-os 5.11.0-7614-generic, i7-9750H CPU @ 2.60GHz 32GB, GeForce GTX 1650 Mobile / Max-Q 4GB

* Yuxuan Zhu
  * [LinkedIn](https://www.linkedin.com/in/andrewyxzhu/)
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1050 4096MB (Personal Laptop)

## Live Online

[![](img/demo.gif)](https://sona1111.github.io/webgpu-super-resolution/)

## Introduction

We created a WebGPU based image super resolution program. Under the hood, it runs a neural netork based on [ESRGAN](https://github.com/xinntao/ESRGAN). The input to the program is any RGB image. The output of the program is the same image with 4x resolution. The program runs in Chrome Canary with the flag `--enable-unsafe-webgpu`. We currently support up-sizing images of size 200px by 200px and size limits may vary depending on different hardwares.

## Credits

* [WebGPU Compute](https://web.dev/gpu-compute/) 
* [WebGPU Samples](https://github.com/austinEng/webgpu-samples)
* [ESRGAN](https://arxiv.org/abs/1809.00219)
* [ESRGAN Weights](https://github.com/xinntao/ESRGAN)
