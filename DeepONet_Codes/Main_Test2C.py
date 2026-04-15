import numpy as onp
import scipy.io
from scipy.interpolate import griddata
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import time
from time import process_time
import jax
import pickle
import pandas as pd
import sys


### global functions ###
from Richards import soil_parameters, VG_model
from MLP import modified_MLP, DataGenerator, architechture
from Test2C_data import generate_one_ics_training_data, generate_one_up_bcs_training_data, generate_one_bcs_training_data, generate_one_res_training_data, generate_flux_data
from Test2C_DeepONet import PI_DeepONet

from jax.lib import xla_bridge
print('You are using a:')
print(xla_bridge.get_backend().platform)


print('This is test 2-scenario(c) Geo1: Flux to solution- water strategy')
############################# BEGIN ####################################################
############################################################################################


soil_type = "loam"

if soil_type == "loam":
    tfin = 1
else:
    tfin = 0.25

print(f"Final time of simulation is = {tfin}")


nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg   = soil_parameters(soil_type)

theta_function, K_function, h_function = VG_model(nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg)

######################### settings ####
#Loss weights
lambdaUPBC = 10
lambdaIC = 1
lambdaBC=1

print(f"Weights used are:")
print(f"lambdaUPBC = {lambdaUPBC}")
print(f"lambdaIC = {lambdaIC}")
print(f"lambdaBC = {lambdaBC}")


max_it= 50 # change it to 200,000 
print("Number of epochs is:", max_it)

print("-" * 120)
# Define the architectures
trunk_depth = 2
print("Trunk depth:", trunk_depth)
trunk_width = 10
branch_depth = 2
print("-" * 120)
print("Branch depth:", trunk_depth)
print("-" * 120)
branch_width = 10
# Define hidden dimension
HD = trunk_width
D = 3 # time-space dimension
print("-" * 120)


batch_size= 50 # batch size of data
print("-" * 120)
print("Batch size used is :", batch_size)
print("-" * 120)


N_train = 80  # number of input samples used for training
print('Number of training input samples',N_train)

m = 10       # number of sensors for each input sample, here in the exp of flux, it means number of ti in (0,T)

P_ics_train = 10  # number of locations for evulating the initial condition
P_up_bcs_train =m  # number of locations for evulating the surface boundary condition
P_bcs_train = 50    # number of locations for evulating lrd boundary condition
P_res_train = 100 # number of locations for evulating the PDE residual


# generate surface water flux data
q0_train = generate_flux_data(N_train, m, soil_type, ksvg)

key = random.PRNGKey(0)
keys = random.split(key, N_train)




# Print a separator line
print("-" * 120)
# Generate training data for PDE residual
u_res_train, y_res_train, s_res_train = vmap(generate_one_res_training_data, in_axes=(0, 0, None, None))(keys, q0_train, m, P_res_train)
u_res_train = u_res_train.reshape(N_train * P_res_train,-1)
y_res_train = y_res_train.reshape(N_train * P_res_train,-1)
s_res_train = s_res_train.reshape(N_train * P_res_train,-1)
print("u res shape", u_res_train.shape)
print("y res shape", y_res_train.shape)
print("s res shape", s_res_train.shape)

# Generate training data for inital condition
u_ics_train, y_ics_train, s_ics_train = vmap(generate_one_ics_training_data, in_axes=(0,0, None))(keys, q0_train, P_ics_train)
u_ics_train = u_ics_train.reshape(P_ics_train*N_train,-1)
y_ics_train = y_ics_train.reshape(P_ics_train*N_train, -1)
s_ics_train = s_ics_train.reshape(P_ics_train*N_train,-1)
print("u ic shape", u_ics_train.shape)
print("y ic shape", y_ics_train.shape)
print("s ic shape", s_ics_train.shape)

# Generate training data for soil surface boundary condition
u_up_bcs_train, y_up_bcs_train, s_up_bcs_train = vmap(generate_one_up_bcs_training_data, in_axes=(0, 0, None, None))(keys, q0_train, m, P_up_bcs_train)
u_up_bcs_train = u_up_bcs_train.reshape(N_train * P_up_bcs_train,-1)
y_up_bcs_train = y_up_bcs_train.reshape(N_train * P_up_bcs_train,-1)
s_up_bcs_train = s_up_bcs_train.reshape(N_train * P_up_bcs_train,-1)
print("u up shape", u_up_bcs_train.shape)
print("y up shape", y_up_bcs_train.shape)
print("s up shape", s_up_bcs_train.shape)

# Generate training data for left right dowm (lrd)boundary condition
u_bcs_train, y_bcs_train, s_bcs_train = vmap(generate_one_bcs_training_data, in_axes=(0, 0, None, None))(keys, q0_train, m, P_bcs_train)
u_bcs_train = u_bcs_train.reshape(N_train * P_bcs_train,-1)
y_bcs_train = y_bcs_train.reshape(N_train * P_bcs_train,-1)
s_bcs_train = s_bcs_train.reshape(N_train * P_bcs_train,-1)
print("u bcs shape", u_bcs_train.shape)
print("y bcs shape", y_bcs_train.shape)
print("s  bcs shape", s_bcs_train.shape)


ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
up_bcs_dataset = DataGenerator(u_up_bcs_train, y_up_bcs_train, s_up_bcs_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

branch_layers = architechture(input_size = m, depth = branch_depth, width = branch_width, output_size = HD)
trunk_layers = architechture(input_size =3, depth = trunk_depth, width = trunk_width, output_size = HD)

print("-" * 120)
print("Branch architechture:", branch_layers)
print("-" * 120)
print("Trunk architechture:", trunk_layers)
print("-" * 120)
###### Initialize model ###########
# Start time measurement
t_start = process_time()

print("Starting model building...")
print("-" * 120)

# Construct the  model
water = PI_DeepONet(branch_layers, trunk_layers, lambdaIC, lambdaBC, lambdaUPBC)

# Stop time measurement
t_stop = process_time()

# Print a separator line
print("-" * 120)

print("Model building completed.")
print("Time taken for model building:", t_stop - t_start, "seconds")
print("-" * 120)

##### training################################################
t_start = process_time()
print("Training process started...")
print("-" * 120)
############################################

water.train(ics_dataset, up_bcs_dataset, bcs_dataset, res_dataset, nIter=max_it)
t_stop = process_time()
print("Training process completed.")
print("Model training duration:", t_stop - t_start, "seconds")
print("-" * 120)
####################################

################### save data #####################

# get model parameters###########################
print("Getting the model parameters...")
params = water.get_params(water.opt_state)
print("-" * 120)
#############################
# Save params to a pickle file
with open(f'{soil_type}_parameters_Test2C_geo1', 'wb') as f:
    pickle.dump(params, f)

#### save the loss functions ###
print("Getting the model loss functions...")
loss_data = pd.DataFrame({'Loss_total': water.loss_log,
    'Loss_ICS': water.loss_ics_log,
    'Loss_BCS': water.loss_bcs_log,
     'Loss_up_BC': water.loss_up_bcs_log,
    'Loss_Res': water.loss_res_log})
loss_data.to_csv(f'{soil_type}_loss_logs_Test2C_geo1.csv', index=False)

### End #####
print("-" * 120)
print("THE END !!! ...")
print("-" * 120)


#### # Read the saved data and create plots in a separate Python script 
############################# END ... ####################################################
############################################################################################