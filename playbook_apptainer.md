# Setting up the environment
NVIDIA [TorchFort](https://github.com/NVIDIA/TorchFort) is a DL training and inference interface for HPC programs. This NVAITC playbook demonstrates how the perform online deep learning traning and inference in an HPC numerical simulation application. Whereas the docker_playbook.ipynb demonstrates how to run the commands interactively using docker exec, this playbook is meant to demonstrate the same workflow on a supercomputer equipped with Slurm and Apptainer. Hence the commands should be executed in the terminal or copy-pasted to a slurm jobscript that is submitted for execution with 'sbatch jobscript.sh'. We also assume that the user has TorchFort container converted to an Apptainer .sif file. User can follow the docker_playbook.ipynb to build the Docker container and CSC provides excellent instructions how to convert it to .sif format (https://docs.csc.fi/computing/containers/creating/#converting-a-docker-container)

As an example application, we use the open-source [CaNS](https://github.com/CaNS-World/CaNS) (Canonical Navier-Stokes) code which can be cloned with the following command on a login node


```python
git clone --recursive https://github.com/CaNS-World/CaNS
```

Subsequently, let's checkout to a state that's been verified to work with the current version of the playbook


```python
cd CaNS && \ 
git checkout de78afb6a62c2c0d785d07b7912216806dc1ade8 && \
cd ..
```

Next, let's validate that we can run the Apptainer container by submitting the following slurm script


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv torchfort.sif nvidia-smi
```

# Online training with TorchFort

As an example case we consider the prediction of wall-bounded turbulence from wall quantities using convolutional neural networks, as presented in ([Guastoni et al. 2020](https://iopscience.iop.org/article/10.1088/1742-6596/1522/1/012022)). In summary, we want to train a PyTorch Neural Network model which takes the wall shear stress field $\mathbf{\tau}(x,y,z,t)$ at the solid wall, $y=y_{wall}$ as an input and predicts the velocity field $\mathbf{v}(x,y,z,t))$ at a certain height $y=y_{pred}$ as an output. 

To achieve this, we need to consider two aspects: 1. Model creation and 2. Application source code-changes to facilitate TorchFort library calls. 

# Model Creation
In the provided /scripts/fcn.py we have implemented a simple Convolutional Neural Network architecture. Let's create the model and save it onto disk as a torchscripted model by submitting the following job. Please note that the number of input and output channel needs to be set according to the task. In this case both $\mathbf{\tau}$ and $\mathbf{u}$ fields have three components so both channel dimensions are set as three. 


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv -B ./files:/files torchfort.sif python /files/python_model/fcn.py
```

# Application source-code changes

There are numerous ways of formulating the online DL training/inference workflow and it will always be application specific. In this section, we detail the necessary building blocks. All of these application source code changes have been implemented for you in /files/CaNS_src_updates/. At the end, we simply just overwrite the original source code with the modified source code.


1. Include TorchFort and CUDA Fortran.

```
  use torchfort
  use cudafor
```

2. Declare and allocate necessary arrays e.g. 

```
  real(rp), allocatable, dimension(:,:,:,:) :: input, output, label
  ...
  allocate( input(1:ng(1), 1:ng(2), 1:3, trainbs), &
           output(1:ng(1), 1:ng(2), 1:3, trainbs), &
            label(1:ng(1), 1:ng(2), 1:3, trainbs))
```

Please note, the example case uses OpenACC for GPU-acceleration. For multi-GPU runs, the simulation is performed using domain decomposition per MPI-process (in z-direction) but training/inference is performed using full $x-z$ planes which are gathered from each subdomain. 

3. Initialise a TorchFort model (or TorchFort distributed for multi-gpu training). Initialisation requires a name identifier for your model, a path to your config file, which specifies hyperparameters, the MPI_communicator for distributed_runs and the local rank of the process. Here you can inspect the example config file [config_fcn_torchscript.yaml](./pencil-code/samples/conv-slab/config_fcn_torchscript.yaml)
```
  if (nranks == 1) then
    istat = torchfort_create_model(model_name, model_config_file, dev)
  else
    istat = torchfort_create_distributed_model(model_name, model_config_file, MPI_COMM_WORLD, dev)
  endif
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
```

4. If you wish to start training from an existing checkpoint we can do it as follows. Please note that each process reads the checkpoint, not just root.

```
  if (torchfort_load_ckpt) then
    if (myid == 0) print*, "Loading torchfort checkpoint", torchfort_ckpt
    istat = torchfort_load_checkpoint(model_name, torchfort_ckpt, isteptrain, istepval)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    if (myid == 0) print*, "isteptrain", isteptrain, "istepval", istepval
  else
    isteptrain = 0
    istepval = 0
  endif
```

The online DL training usually happens *within* the main simulation time/iteration loop and this is the case for the example. ALL of the following blocks of code are within this time loop scope. The time loop starts with the following line.
```
  do while(.not.is_done)
```

5. If we train, we must copy the local solution and label states to the buffers and distribute them, call the training method and optionally print out the loss value
```
    if (is_training .and. (timesample >= dtsample_scaled)) then
       ...
      !$acc kernels default(present)
       ! copy inputs
       input_local(1:n(1), 1:n(2), 1, sampleidx) = (tauxz(1:n(1),1:n(2),0) - txzmean(1)) / txzstd(1)
       input_local(1:n(1), 1:n(2), 2, sampleidx) = (tauyz(1:n(1),1:n(2),0) - tyzmean(1)) / tyzstd(1)
       input_local(1:n(1), 1:n(2), 3, sampleidx) = (tauzz(1:n(1),1:n(2),0) - tzzmean(1)) / tzzstd(1)
       ! copy labels
       label_local(1:n(1), 1:n(2), 1, sampleidx) = (u(1:n(1),1:n(2), kbot) - umean(kbot)) / ustd(kbot)
       label_local(1:n(1), 1:n(2), 2, sampleidx) = (v(1:n(1),1:n(2), kbot) - vmean(kbot)) / vstd(kbot)
       label_local(1:n(1), 1:n(2), 3, sampleidx) = (w(1:n(1),1:n(2), kbot) - wmean(kbot)) / wstd(kbot)
       !$acc end kernels

       if (mod(sampleidx, trainbs * nranks) == 0) then
           sampleidx = 0
           call distribute_batches(input_local, input, n, ng)
           call distribute_batches(label_local, label, n, ng)

           if (.not. is_validating) then
             isteptrain = isteptrain + 1
             !$acc host_data use_device(input, label)
             istat = torchfort_train(model_name, input, label, loss_value)
             !$acc end host_data
             if (istat /= TORCHFORT_RESULT_SUCCESS) stop

             if (myid == 0) print*, "Training step", isteptrain, "training loss = ", loss_value

```
However, please note that the memory copying is only necessary due to the nature of the case. If we were to train a model that uses the global (or subdomain) solution state as an input we could directly pass it to the train method.

6. If we perform inference, we follow the same structure of copying the local solution state, distributing the local states to form a global state and then performing the inference 
```
       if (test_inference) then
          !$acc kernels default(present)
          input_local(1:n(1), 1:n(2), 1, sampleidx) = (tauxz(1:n(1),1:n(2),0) - txzmean(1)) / txzstd(1)
          input_local(1:n(1), 1:n(2), 2, sampleidx) = (tauyz(1:n(1),1:n(2),0) - tyzmean(1)) / tyzstd(1)
          input_local(1:n(1), 1:n(2), 3, sampleidx) = (tauzz(1:n(1),1:n(2),0) - tzzmean(1)) / tzzstd(1)
          !$acc end kernels

          if (mod(sampleidx, trainbs * nranks) == 0) then
             sampleidx = 0
             call distribute_batches(input_local, input, n, ng)

             !$acc host_data use_device(input, output)
             istat = torchfort_inference(model_name, input, output)
             !$acc end host_data
             if (istat /= TORCHFORT_RESULT_SUCCESS) stop
```
8. Checkpoints can be onto disk as follows. Please note that only the root rank will do the save.
```
       if (myid == 0) then
         write(trainckptnum,'(i7.7)') isteptrain
         filename = 'torchfort_checkpoint_'//trainckptnum
         print "(a20,i10,a12,a100)", "Writing checkpoint ", isteptrain, " directory = ", filename
         istat = torchfort_save_checkpoint(model_name, filename)
         if (istat /= TORCHFORT_RESULT_SUCCESS) stop
       endif
```

Please note that data-normalisation with the online training approach can be challenging. In this case we use

$\hat{x} = \frac{x - \langle x \rangle}{\langle (x - \langle x \rangle)^2 \rangle}$

where the mean and standard deviation have are acquired by pre-running (developing the flow from the initial conditions) the case without any training.

Finally, let's copy the modified source code into CaNS with the following command on a login-node


```python
cp ./files/CaNS_src_updates/* ./CaNS/src/
```

# COMPILATION
Code building process will of course vary between applications. Most of the times it is sufficient simply to include and link TorchFort with
```
-L$(TORCHFORT_ROOT)/lib -ltorchfort_fort -ltorchfort -L$(HDF5_ROOT)
-I$(TORCHFORT_ROOT)/include -I$(HDF5_ROOT)
```
In this case, we provide slighly modified Makefiles and configs that replace those of the Original CaNS repo. Please execute these on a login-node


```python
cp ./files/CaNS_config_updates/Makefile ./CaNS/Makefile &&
cp ./files/CaNS_config_updates/build.conf ./CaNS/build.conf &&
cp ./files/CaNS_config_updates/flags.mk ./CaNS/configs/flags.mk &&
cp ./files/CaNS_config_updates/external.mk ./CaNS/dependencies/external.mk
```

Remember that the idea is to bring the application into the container where it can be easily compiled against pre-built TorchFort with the NVHPC SDK. Hence, the we execute compilation within the Apptainer container by submitting the following job


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv -B ./CaNS:/opt/CaNS -B ./files:/files torchfort.sif bash -c "cd /opt/CaNS && make libs && make"
```

# Test case setup

Now that the TorchFort-enabled CaNS code has been build, the last step is to run the test case. First, let's copy the previously generated CNN model to the case folder by executing the following on a login-node


```python
cp ./files/python_model/cans_fcn.pt ./files/reconstruction_case
```

CaNS code will read runtime parameters from an input file 'input.nml' located in the case folder. This example comes with two CaNS input files: 1. input.stats.nml which is used to run the case up to 2000 time units without training to develop the flow and measure the normalisation statistics and 2. input.train.nml to start training from 2000 time units onwards. Let's first run the first part. Please note that we have to add the path to CaNS dependencies to LD_LIBRARY_PATH environment variable as they are not set in the container build.


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv -B ./CaNS:/opt/CaNS -B ./files:/files torchfort.sif bash -c 'export LD_LIBRARY_PATH=/opt/CaNS/dependencies/cuDecomp/build/lib:${LD_LIBRARY_PATH} && \
                                                                                 cd /files/reconstruction_case && \
                                                                                 cp input.stats.nml input.nml && \
                                                                                 mpirun -np 1 --allow-run-as-root --bind-to none /opt/CaNS/run/cans'
```

Now that we have a developed checkpoint (to exclude the initial transients) for the simulation and an estimate of the flow statistics for normalisation, we can start the training. We will overwrite the input.nml with it's training counterpart input.train.nml which specifies all necessary runtime parameters for traing e.g.
```
trainbs = 32
nsamples_train = 3200
nsamples_val = 320
```
which means that we will undertake nsamples_train/(trainbs * num_gpus) number of training steps, after which we will take nsamples_val/(trainbs * num_gpus) validation steps. Training checkpoints and inference results are save after each validation epoch. For the purposes of this demo the training step will proceed for a total of 500 steps. However, for best results the training should run considerably longer.


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv -B ./CaNS:/opt/CaNS -B ./files:/files torchfort.sif bash  -c 'export LD_LIBRARY_PATH=/opt/CaNS/dependencies/cuDecomp/build/lib:${LD_LIBRARY_PATH} && \
                                                                                  cd /files/reconstruction_case && \
                                                                                  cp input.train.nml input.nml && \
                                                                                  mpirun -np 1 --allow-run-as-root --bind-to none /opt/CaNS/run/cans'
```

# Plot results

Let's now plot the results. We can use this python script to do the plotting within the container. Simply create plotter.py file, copy-paste the contents, and submit the plotting job with the following script.


```python
import h5py
import matplotlib.pyplot as plt

data = h5py.File("/files/reconstruction_case/data/sample_0000500_0000.h5", "r")
print(list(data))
print(data['label'].shape)

fig = plt.figure(figsize=(7, 10))
for i, var in enumerate(['u', 'v', 'w']):
    plt.subplot(3, 2, i*2 + 1)
    plt.imshow(data["pred"][0,i,:,:])
    plt.colorbar()
    plt.title(var)

    plt.subplot(3, 2, i*2 + 2)
    plt.imshow(data["label"][0,i,:,:])
    plt.colorbar()
    plt.title(var)

plt.savefig("results.pdf")
```


```python
#!/bin/bash
#SBATCH --job-name=test_torchfort
#SBATCH --account=12345(this is cluster-specific)
#SBATCH --partition=gpu(this is cluster-specific)
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1(this is cluster-specific)
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out

apptainer exec --nv -B ./plotter.py:/opt/plotter.py -B ./files:/files torchfort.sif python /opt/plotter.py
```
