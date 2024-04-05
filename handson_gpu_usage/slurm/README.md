# Steps to run jobs on GPUs using slurm.

1. ssh to gorina11.
  a. ssh to ssh3.ux.uis.no first if you are outside UiS network. 
3. run `sbatch conda_setup.sh`
4. Monitor the conda env creation using `tail -f conda_setup.out`
5. Once the conda env is successfully created and all packages are installed, then run `sbatch imdb_train.sh`
6. Monitor the logs `tail -f imdb_train.out` or in wandb.ai
