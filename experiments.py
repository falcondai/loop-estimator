import numpy as np


from mrp import RiverSwim, MdpPi
from mc import n2p, p2n
import estimate


def extract_last_update(v_hats, times, sls):
    last_time_idx = 0
    last_time = times[last_time_idx]
    sl = sls[0]
    while times[last_time_idx + 1] <= sl:
        

if __name__ == '__main__':
    # print(estimate.model_based(6, [], 0.9))
    # print(estimate.model_based(6, [(1, 1)], 0.9))
    # mdp = RiverSwim()
    # n_states = mdp.n_states
    # # The move-right policy
    # pi = [1] * mdp.n_states
    # mrp = MdpPi(mdp, pi)
    # # print(mrp.p)
    # # print(mrp.r)
    # n = p2n(mrp.p)
    # print(n)
    # print(n.diagonal())
    #
    # # Experiment
    # n_steps = 10 ** 5
    # discount = 0.999
    # print('length of trajectory', n_steps)
    # print('discount', discount)
    # v = exact(mrp.p, mrp.r, discount)
    # print('exact', v)
    #
    # samples = [(s, r) for (s, r) in mrp.sample_transition(n_steps)]
    #
    # v_hat = geometric(n_states, samples, discount)
    # print('geometric', np.max(np.abs(v - v_hat)), v_hat)
    #
    # v_hat = truncate(n_states, samples, 100, discount)
    # print('truncate', np.max(np.abs(v - v_hat)), v_hat)
    #
    # # alpha_func = lambda n: 1 / np.sqrt(1 + n)
    # alpha_func = lambda n: 1 / (1 + n)
    # # alpha_func = lambda n: 0.1
    # v_hat = td(n_states, samples, alpha_func, discount)
    # print('td', np.max(np.abs(v - v_hat)), v_hat)
    #
    # v_hat = loop(n_states, samples, discount)
    # print('loop', np.max(np.abs(v - v_hat)), v_hat)
    #
    # v_hat = model_based(n_states, samples, discount)
    # print('model_based', np.max(np.abs(v - v_hat)), v_hat)

    rs = RiverSwim()
    pi = [1] * 6
    mrp = MdpPi(rs, pi)
    # from pylab import *
    discounts = [0.9, 0.99, 0.999, 0.9999]
    # discounts = [0.9]
    n_trials = 1
    n_traj_len = 100000
    # n_traj_len = 30000
    step = 10000
    min_step = 0
    sls = np.arange(min_step, n_traj_len, step)
    geom_seeds = [123, 99, 23, 94538, 5943, 3471]
    # all_td_steps = [1, 10, 100, 1000]
    all_td_steps = [1, 10]
    all_tds = [np.zeros((n_trials, len(sls), 6)) for _ in all_td_steps]
    v_hat_geoms = np.zeros((n_trials, len(sls), 6))
    v_hat_loops = np.zeros((n_trials, len(sls), 6))
    v_hat_mbs = np.zeros((n_trials, len(sls), 6))
    # legends = ['loop', 'geom', 'model'] + ['td(%i)' % td_step for td_step in all_td_steps]

    # fig, axes = plt.subplots(len(discounts), 1, sharex=True, figsize=(12, 24), dpi=80)
    # print(axes)

    for discount in discounts:
        # plt.subplot(ax)
        v = estimate.exact(mrp.p, mrp.r, discount)
        for tr in range(n_trials):
            # Generate a sample path
            mrp.state = 0
            s_r = [(s, r) for s, r in mrp.sample_transition(n_traj_len)]
            head_t = 0
            # States
            mb_state = None

            for i in range(6):
                # v_hat_geoms[tr, j, i]
                _, v_geoms, times = estimate.geometric_single_times(i, s_r, discount, seed=geom_seeds[i] + tr * 17)


            for j, sl in enumerate(sls):
                new_segment = s_r[head_t:sl]
                # for i in range(6):
                #     v_hat_loops[tr, j, i] = estimate.loop_single(i, s_r[:sl], discount)[0]
                #     v_hat_geoms[tr, j, i] = estimate.geometric_single(i, s_r[:sl], discount, seed=geom_seeds[i] + tr * 17)[0]
                v_hat_mbs[tr, j], mb_state = estimate.model_based(6, new_segment, discount, state=mb_state)

                # for k, td_step in enumerate(all_td_steps):
                #     all_tds[k][tr, j] = estimate.td_k(td_step, 6, s_r[:sl], discount)

                # Advance head
                head_t = sl

        v_hats_all = [v_hat_loops, v_hat_geoms, v_hat_mbs] + all_tds

    #     # Plot by inf-norm
    #     for name, v_hats in zip(legends, v_hats_all):
    #         dd = np.abs(v_hats - v.reshape((1, 1, -1))).max(2)
    #         dmean = dd.mean(0)
    #         derr = dd.std(0)
    #         plot(sls, dmean, 'x-', label=name)
    #         fill_between(sls, dmean - derr, dmean + derr, alpha=0.2)
    #
    #     yscale('log')
    #     ylabel('$\infty$-norm error')
    #     title('$\gamma=%f$' % discount)
    #
    # xlim(left=min_step, right=sls[-1])
    # xlabel('steps $T$')
    # legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=len(legends))
    #
    # savefig('gammas_plot_%i.pdf' % n_trials)
