`$ conda env create -f environment.yml`

`$ conda activate InvPendEnv`

## Gymnasium Environment
The gymnasium environment must be installed as a package with pip before it can be used.
```sh
pip install -e ./inv_pendulum_env
```
You can then use it in python:
```py
import gymnasium
import inv_pend_env

env = gymnasium.make("inv_pend_env/inv_pendulum_v0")
```
The following defaults can be specified:
```py
render_mode=None, 
setpoint: int | float=0,
length: int | float=1, 
mass: int | float=1, 
gravity: int | float=9.81,
plot: bool = False, 
seed: int | float = None, 
disallowcontrol: bool = False, 
timestep: int | float = 0.1,
terminate: bool = True,
```


## Model code

The model code is in model-sb3.py. In order to use it you must first train the model, which can be done by specifying the length of training in the code (default is 500,000) and running `python model-sb3.py train`. Once this has completed it is possible to use the model or a PID controller to obtain a variety of different graphs. The easiest graphs are the eval graphs, designed to observe how the model was performing quickly after training. There is a success graph which graphs the successful areas using the top left corner, and there is a force graph which shows the force by command. these graphs can be made by running `python model-sb3.py eval success` and `python model-sb3 eval initforce` respectively. There is also a test subcommand which will run a visual simulation of the inverted pendulum so that you can watch the model in action. This can be rin with `python model-sb3.py test`. Finally, there is the verification code. This can either show a box or verify areas on the graph. To run this you run python `model-sb3.py verify` and add `box` to simulate the effect on a box, `quant` to quantize the inputs, or `pid` if you wish to use a pid controller