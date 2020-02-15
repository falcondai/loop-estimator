import numpy as np


def exact(p, r, discount):
    '''Solve for the exact values via matrix inversion.'''
    n_states = p.shape[0]
    v = np.linalg.inv(np.eye(n_states) - discount * p) @ r.reshape((-1, 1))
    return v.reshape(-1)


def co_td_k(k, n_states, discount, eta_func):
    '''
    Co-routine implementation of TD(k) with arbitrary learning rate.
    Args:
        eta_func : Nat -> Real+.
            Learning rates of TD(k). It will be passed the number of visits of a given state. The classic choice is `t |-> 1 / t` which satisfies the Robbins-Monro condition.
    '''
    # Init
    v_hat = np.zeros(n_states)
    visit = np.zeros(n_states)
    gammas = discount ** np.arange(k + 1)
    history = []
    yield
    while True:
        # XXX the head is updated by partial sum of (k + 1) rewards, including its own and ends with the next state
        # There are (k + 2) items
        # Consider TD(0)
        if len(history) <= k + 2:
            # Yield estimates
            # Receive transitions
            transitions = yield np.array(v_hat)
            history += transitions
        else:
            # Consume transitions
            head_return = np.sum(gammas * np.array([r for _, r in history[:k + 1]]))
            x, _ = history.pop(0)
            y, _ = history[k + 1]
            update = head_return + (discount ** (k + 1)) * v_hat[y]
            visit[x] += 1
            eta = eta_func(visit[x])
            v_hat[x] = eta * update + (1 - eta) * v_hat[x]


def co_loop_single(state, discount):
    '''
    Co-routine implementation of loop estimator for a single state.
    Args:
        state : Int.
            The state whose value to estimate.
    '''
    # Init
    yield
    lr_hat = 0
    ld_hat = 0
    partial_return = 0
    loop_count = 0
    ago = None
    history = []
    while True:
        if len(history) < 2:
            # Yield estimates and receive new transitions
            # Plug in estimates
            loop_estimate = lr_hat / (1 - ld_hat)
            transitions = yield loop_estimate
            history += transitions
        else:
            # Consume transitions
            x, r = history.pop(0)
            y, _ = history[0]
            if x == state:
                # Start of a loop, reset accumulators
                loop_count += 1
                ago = 0
                partial_return = 0
            if ago is not None:
                partial_return += discount ** ago * r
                ago += 1
                if y == state:
                    # End of a loop, update estimates
                    eta = 1 / loop_count
                    lr_hat = eta * partial_return + (1 - eta) * lr_hat
                    ld_hat = eta * discount ** ago + (1 - eta) * ld_hat


def co_loop(n_states, discount):
    '''Co-routine implementation of loop estimator for all states by running a copy of `co_loop_single` for each state.'''
    yield
    gen = [co_loop_single(state, discount) for state in range(n_states)]
    last_estimate = np.zeros(n_states)
    for s in range(n_states):
        gen[s].send(None)
        last_estimate[s] = gen[s].send([])
    while True:
        transitions = yield np.array(last_estimate)
        for s in range(n_states):
            last_estimate[s] = gen[s].send(transitions)


def co_model_based(n_states, discount):
    '''Co-routine implementation of the model-based estimator with add-1 smoothing.'''
    # Init
    yield
    # Add-1 smoothing for transitions
    r_hat = np.zeros(n_states)
    visit = np.ones((n_states, n_states)) / n_states
    history = []
    while True:
        if len(history) <= 2:
            # Yield estimates and receive new transitions
            p_hat = visit / visit.sum(1, keepdims=True)
            # Plug in estimates
            transitions = yield exact(p_hat, r_hat, discount)
            history += transitions
        else:
            # Consume transitions
            x, r = history.pop(0)
            y, _ = history[0]
            visit[x, y] += 1
            eta = 1 / visit[x].sum()
            r_hat[x] = eta * r + (1 - eta) * r_hat[x]
