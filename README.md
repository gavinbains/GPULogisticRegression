# GPULogisticRegression

## Running CUDA code:
Putting instructions in here since I'm tired of looking at a prior blackboard assignment to figure this out.
1. Login to HPC w/ USC credentials at @ hpc-login3.usc.edu
2. Setup the CUDA toolchain with `source /usr/usc/cuda/default/setup.sh` from your home directory. *Note*: This has to be done every time you login to HPC, I don't know why.
3. Compile your program with the CUDA compiler: `nvcc -o <executable name> -O3 <filename>`.
4. Run your executable on the *GPU* using `srun -n1 --gres=gpu:1 ./<executable name>`. *Note*: Please do not do just `./<executable name>`. It will work but it will not run on the GPU making all our CUDA code effectively useless and will keep you thinking that your code is wrong when it really might not be.
