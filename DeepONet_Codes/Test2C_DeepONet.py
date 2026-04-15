from functools import partial
from MLP import modified_MLP, DataGenerator
from jax import random, grad, vmap, jit
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
import itertools
from Richards import soil_parameters, VG_model
from tqdm import trange


soil_type = "loam"

nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg   = soil_parameters(soil_type)

theta_function, K_function, h_function = VG_model(nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg)

# Physics-informed DeepONet
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers, lambdaIC, lambdaBC, lambdaUPBC):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = modified_MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = modified_MLP(trunk_layers, activation=np.tanh)

        self.lambdaIC, self.lambdaBC, self.lambdaUPBC  =  lambdaIC, lambdaBC, lambdaUPBC

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                      decay_steps=1000,
                                                                      decay_rate=0.95))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_up_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, t, x, z):
        branch_params, trunk_params = params
        y = np.stack([t,x,z])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return -np.exp(outputs)

    # Define ds/dx
    def s_grad(self, params, u, t, x, z):
         s_t = grad(self.operator_net, argnums=2)(params, u, t, x, z)
         s_x = grad(self.operator_net, argnums=3)(params, u, t, x, z)
         s_z = grad(self.operator_net, argnums=4)(params, u, t, x, z)
         s_xx= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, t, x, z)
         s_zz= grad(grad(self.operator_net, argnums=4), argnums=4)(params, u, t, x, z)
         return s_t, s_x, s_z, s_xx, s_zz



    # Define PDE residual
    def residual_net(self, params, u, t, x, z):
        s =  self.operator_net(params, u, t, x, z)
        dtheta_ds= grad(theta_function)(s)
        K = K_function(s)
        dK_ds =   grad(K_function)(s)
        s_t, s_x, s_z, s_xx, s_zz = self.s_grad(params, u, t, x, z)
        res = dtheta_ds*s_t - dK_ds *(s_x**2 +s_z**2) - K*(s_xx+s_zz) - dK_ds * s_z
        return res

    # initial loss
    def loss_ics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, y[:,0], y[:,1], y[:,2])
        theta = theta_function(s)

        # Compute loss
        loss = np.mean((outputs.flatten() - theta)**2)
        return loss

    # upper boundary loss
    def loss_up_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, y[:,0], y[:,1], y[:,2])
        _, _, s_z, _, _ = vmap(self.s_grad, (None, 0, 0, 0, 0))(params, u, y[:,0], y[:,1], y[:,2])
        K = K_function(s)
        q = -K*(s_z+1)

        loss_up = np.mean((outputs.flatten() - q)**2)

        return loss_up

    # l-r-d boundary loss
    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Left
        i=0
        s = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        K = K_function(s)
        _, s_x, _, _, _ = vmap(self.s_grad, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        loss_left = np.mean((K*s_x)**2)

        # Right
        i=3
        s = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        K = K_function(s)
        _, s_x, _, _, _ = vmap(self.s_grad, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        loss_right = np.mean((K*s_x)**2)

        # down
        i=6
        s = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        _, _, s_z, _, _ = vmap(self.s_grad, (None, 0, 0, 0, 0))(params, u, y[:,i], y[:,i+1], y[:,i+2])
        loss_down = np.mean(s_z**2)


        return loss_left + loss_right + loss_down

    # residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        res = vmap(self.residual_net, (None, 0, 0, 0, 0))(params, u, y[:,0], y[:,1], y[:,2])

        # Compute loss
        loss = np.mean(res**2)
        return loss

    # total loss
    def loss(self, params, ics_batch, up_bcs_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_up_bcs = self.loss_up_bcs(params, up_bcs_batch)
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = self.lambdaIC*loss_ics + self.lambdaUPBC*loss_up_bcs+ self.lambdaBC*loss_bcs +  loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, up_bcs_batch, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ics_batch, up_bcs_batch, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, ics_dataset, up_bcs_dataset, bcs_dataset, res_dataset, nIter = 1000):
        ics_data = iter(ics_dataset)
        up_bcs_data = iter(up_bcs_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            ics_batch= next(ics_data)
            up_bcs_batch= next(up_bcs_data)
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, up_bcs_batch, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ics_batch, up_bcs_batch, bcs_batch, res_batch)
                loss_ics_value = self.loss_ics(params, ics_batch)
                loss_up_bcs_value = self.loss_up_bcs(params, up_bcs_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_up_bcs_log.append(loss_up_bcs_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_ics' : loss_ics_value,
                                   'loss_up_bcs' : loss_up_bcs_value,
                                  'loss_bcs' : loss_bcs_value,
                                  'loss_physics': loss_res_value})


    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1],  Y_star[:,3])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1],  Y_star[:,3])
        return r_pred
    
