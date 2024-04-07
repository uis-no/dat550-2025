# Steps to run jobs on GPUs using slurm.

1. ssh to gorina11.
  - ssh to ssh3.ux.uis.no first if you are outside UiS network.
  - ssh to gorina11.ux.uis.no
2. run `sbatch conda_setup.sh`
3. Monitor the conda env creation using `tail -f conda_setup.out`
4. Once the conda env is successfully created and all packages are installed, then run `sbatch imdb_train.sh`
5. Monitor the logs `tail -f imdb_train.out` or in wandb.ai
