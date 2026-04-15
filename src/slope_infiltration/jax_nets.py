from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random


def modified_mlp(layers, activation=jnp.tanh):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.0)
        weights = glorot_stddev * random.normal(key, (d_in, d_out))
        bias = jnp.zeros(d_out)
        return weights, bias

    def init(rng_key):
        u1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        u2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            w, b = xavier_init(key, d_in, d_out)
            return w, b

        _, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params, u1, b1, u2, b2

    def apply(params, inputs):
        params, u1, b1, u2, b2 = params
        u = activation(jnp.dot(inputs, u1) + b1)
        v = activation(jnp.dot(inputs, u2) + b2)

        state = inputs
        for w, b in params[:-1]:
            outputs = activation(jnp.dot(state, w) + b)
            state = outputs * u + (1.0 - outputs) * v

        w, b = params[-1]
        return jnp.dot(state, w) + b

    return init, apply


def architecture(input_size: int, depth: int, width: int, output_size: int) -> list[int]:
    return np.concatenate([[input_size], width * np.ones(depth), [output_size]]).astype(int).tolist()
