from jax import random
import numpy as onp
import jax.numpy as np
import pickle

# Geneate ics training data corresponding to one input sample
def generate_one_ics_training_data(key, u0, P=101):
      subkeys = random.split(key, 2)
      t_ic = np.tile(0, (P, 1))
      x_ic = random.uniform(subkeys[0], (P,1))
      z_ic =  random.uniform(subkeys[1], (P,1))
      y = np.hstack([t_ic, x_ic, z_ic])
      u = np.tile(u0, (P, 1))
      s = np.tile(0.2, (P, 1))
      return u, y, s

# Geneate  soil surface bcs training data corresponding to one input sample
def generate_one_up_bcs_training_data(key, u0, m=101, P=100, tfin=1):
    subkeys = random.split(key, 2)
    t_bc =  np.linspace(0,tfin,P)[:, None]
    x_bc = random.uniform(subkeys[1], (P,1))
    z_bc = np.tile(1., (P, 1))

    y =  np.hstack([t_bc, x_bc, z_bc])
    u = np.tile(u0, (P, 1))
    s = u0

    return u, y, s

# Geneate other bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, m=101, P=100, tfin=1):
    subkeys = random.split(key, 6)
    # left
    t_left = tfin*random.uniform(subkeys[0], (P,1))
    x_left = np.tile(0., (P, 1))
    z_left = random.uniform(subkeys[1], (P,1))

    # right
    t_right = tfin*random.uniform(subkeys[2], (P,1))
    x_right = np.tile(1., (P, 1))
    z_right = random.uniform(subkeys[3], (P,1))

    # down
    t_down = tfin*random.uniform(subkeys[4], (P,1))
    x_down = random.uniform(subkeys[5], (P,1))
    z_down = np.tile(0., (P, 1))

    y_left = np.hstack([t_left, x_left, z_left])
    y_right = np.hstack([t_right, x_right, z_right])
    y_down =np.hstack([t_down, x_down, z_down])


    y =  np.hstack([y_left, y_right, y_down])
    u = np.tile(u0, (P, 1))
    s = np.zeros((P, 1))

    return u, y, s


# Geneate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, m=101, P=101, tfin=1):

    subkeys = random.split(key, 3)

    t_res = tfin*random.uniform(subkeys[0], (P,1))
    x_res = random.uniform(subkeys[1], (P,1))
    z_res = random.uniform(subkeys[2], (P,1))

    u = np.tile(u0, (P, 1))
    y =  np.hstack([t_res, x_res, z_res])
    s = np.zeros((P, 1))
    return u, y, s


def generate_flux_data(N_train, m, soil_type, ksvg, tfin=1):
    #print('Generating Training data: Water flux')
    t = np.linspace(0, tfin, m)
    ti = np.array([0, 0.2, 0.5, 0.7, 1])
    q0_train =  np.zeros((N_train, m))

    for i in range(N_train):
        closest_index_ex = 0
        for j in range(len(ti)-1):
            closest_index = np.abs(t - ti[j + 1]).argmin()
            q0_train = q0_train.at[i, closest_index_ex:closest_index+1].set(onp.random.uniform(-0.25*ksvg, 0.00024))
            closest_index_ex = closest_index
    # save data with pickle
    with open(f'{soil_type}TrainFlux_Test2C_geo1', 'wb') as f:
        pickle.dump(q0_train, f)

    return q0_train