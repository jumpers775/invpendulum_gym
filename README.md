# Setting up the Environment

First install [miniconda](https://docs.anaconda.com/miniconda/), then run the following commands to create and enter the environment:

```sh
$ conda env create -f environment.yml
$ conda activate InvPendEnv
```

# Gymnasium Environment
The gymnasium environment must be installed as a package with pip before it can be used.
```sh
$ pip install -e ./inv_pendulum_env
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


# Making and Using the Model

## PID controller

A PID controller is included in this code as it is the easiest solution to this problem and it can be used to compare against model performance. This can be used on any graph instead of the neural network by appending `pid` to the command

## Training

In order to use the model, you must first train it. This can be done by running the following.

```sh
$ python model-sb3.py train
```
This will train the model for 500,000 steps and output it to `/path/to/invpendulum_gym/checkpoints/model-sb3.pth`. In my testing 2-4 million steps resulted in an effective model. To train for more steps you can modify the code to specify a different training duration, or you can run the following to train for a further 500,000 steps.
```sh
$ python model-sb3.py train continue
```

## Testing Model Performance

There are 2 seperate graphs which can be generated to test the performance of the model. The first one will run the model on a set of starting conditions to test when the model will succeed. This can be generated by running the following:
```sh
$ python model-sb3.py eval success
```
You can also generate a graph of the command by condition to observe how the model will react to different scenarios. This can be generated by the following:
```sh
$ python model-sb3.py eval initforce
```


## Quantization

Verification of the model directly is impossible as it can take any floating point number and it is impossible to test them all. For this reason quantization is used. To quantize the model we create many different regions within the conditions and round every input to the center of its region. This results in a testable number of centers whoch can get passed through the model, allowing for verification. This can be enabled in any of the verification graphs by appending `quant` to the command.

## Verification

To verify the model we first break the possible conditions up into regions (the same ones used for quantization). We plot points on the edges of each region and run them through the model (using the command for the center if quantization is in use). Once this has been done you can check if any points in the transformed region would fail or would enter any regions which could fail. Once you remove all regions that can fail you are left with a collection of regions which are provably safe.

To observe the transformation of one region (region size and whch region to use can be specified by modifying the code):
```sh
$ python model-sb3.py verify box
```
To observe which regions are provably safe (region size can be set by modifying the code):
```sh
$ python model-sb3.py verify
```
