# NVAITC Playbook for HPC-AI online training approaches

Online deep learning training and inference are terms used to describe an approach where a numerical HPC simulation and DL training/inference processes are tightly coupled. This NVAITC playbook demonstrates how the perform online deep learning training and inference in an HPC numerical simulation application using NVIDIA's TorchFort library. 

## Getting started

We provide the playbook in two formats:
1. playbook_docker.ipynb is meant to be run interactively on a system which contains docker and jupyter-lab.
Simply start jupyter lab in the working directory as
```
jupyter-lab --port <port> 
```
and copy the URL to your browser to get started.

2. playbook_apptainer.md is meant to demonstrate the workflow on a supercomputer equipped with Slurm and Apptainer.
The specified commands should be executed in the (login node) terminal or copy-pasted to a slurm jobscript that is submitted for execution with 'sbatch jobscript.sh'

## Acknowledgements
TorchFort and the basis of the reconstruction case have been developed by Josh Romero and Thorsten Kurth.
