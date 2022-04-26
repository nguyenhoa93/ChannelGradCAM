cd $HOME/projects
srun -J "channel" --pty --gres=gpu:1 jupyter notebook --no-browser --ip="0.0.0.0"