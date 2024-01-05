### (Double) Deep Q-Learning

If you want to start training from scratch, make sure that `use_savepoint` is set to `False` in `main.py`.

If you want to continue training from a previously trained model, set it to `True`. It will then load the model and the
replay memory from the `savepoints/DeepQLearning` directory.
Savepoints are automatically created every 100 episodes.
The current savepoint has learned for 2400 episodes.

> Warning: If you don't want to overwrite your previous savepoints, make sure to move the `DeepQLearning` folder and its
> contents to a different location, because it will just overwrite the contents of `savepoints`.
>
> Alternatively, you can change the basepaths in `main.py` where to save to and where to load from.

Graphs of the training progress are generated every 50 episodes and saved to the `screenshots` directory.

> Warning: If you don't want to overwrite your previous screenshots when starting a new run, make sure to copy them to a
> different location.
