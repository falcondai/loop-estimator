import numpy as np


from mrp import RiverSwim, MdpPi
from mc import n2p, p2n
from estimate import geometric, truncate, td, loop, model_based, exact


if __name__ == '__main__':
    mdp = RiverSwim()
    n_states = mdp.n_states
    # The move-right policy
    pi = [1] * mdp.n_states
    mrp = MdpPi(mdp, pi)
    # print(mrp.p)
    # print(mrp.r)
    n = p2n(mrp.p)
    print(n)
    print(n.diagonal())

    # Experiment
    n_steps = 10 ** 5
    discount = 0.999
    print('length of trajectory', n_steps)
    print('discount', discount)
    v = exact(mrp.p, mrp.r, discount)
    print('exact', v)

    samples = [(s, r) for (s, r) in mrp.sample_transition(n_steps)]

    v_hat = geometric(n_states, samples, discount)
    print('geometric', np.max(np.abs(v - v_hat)), v_hat)

    v_hat = truncate(n_states, samples, 100, discount)
    print('truncate', np.max(np.abs(v - v_hat)), v_hat)

    # alpha_func = lambda n: 1 / np.sqrt(1 + n)
    alpha_func = lambda n: 1 / (1 + n)
    # alpha_func = lambda n: 0.1
    v_hat = td(n_states, samples, alpha_func, discount)
    print('td', np.max(np.abs(v - v_hat)), v_hat)

    v_hat = loop(n_states, samples, discount)
    print('loop', np.max(np.abs(v - v_hat)), v_hat)

    v_hat = model_based(n_states, samples, discount)
    print('model_based', np.max(np.abs(v - v_hat)), v_hat)
