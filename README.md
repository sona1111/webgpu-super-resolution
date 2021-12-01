# webgpu-super-resolution

We need to consider the different ways of storing the matrix
batch_size * channel_size * height * width to mirror pytorch
only support batchsize of 1 rn
There is a potential bug of i32 and u32 for global id
Does wgsl support loop unrolling?
Webgpu documntation incomplete for command encoders pipelines etc.
Add package.json
Put the loop in GPU
minimize memmory pingpong between cpu and gpu

Performance Analysis
Image size vs Run time
Webgpu vs pytorch on same machine

Wrapper for 2d convolution
interpolation layer 
relu layer with webgpu

Use the same buffer/uniform for matrix size?

--max_old_space_size=30000

