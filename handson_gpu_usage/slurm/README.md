# Steps to run jobs on GPUs using slurm.

1. ssh to gorina11.
  - ssh to ssh3.ux.uis.no first if you are outside UiS network.
  - ssh to gorina11.ux.uis.no
4. Run `sbatch imdb_train.sh`
5. Monitor the logs `tail -f imdb_train.out` or in wandb.ai
