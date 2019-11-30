from collections import defaultdict
import numpy as np

from mrp import MarkovRewardProcess


def exact(p, r, discount):
    '''Solve for the exact values via matrix inversion. Good for a few states.'''
    n_states = p.shape[0]
    v = np.linalg.inv(np.eye(n_states) - discount * p) @ r.reshape((-1, 1))
    return v.reshape(-1)


def truncate(n_states, transitions, k, discount):
    '''Truncate the first k steps and take the discounted sum.'''
    gamma = discount ** np.arange(k)
    visit = np.zeros(n_states)
    empirical_mean = np.zeros(n_states)
    for t, (state, reward) in enumerate(transitions[:-k]):
        ss, rr = zip(*transitions[t:t + k])
        discounted_tot = sum(rr * gamma)
        empirical_mean[state] = (visit[state] * empirical_mean[state] + discounted_tot) / (visit[state] + 1)
        visit[state] += 1
    return empirical_mean


def geometric(n_states, transitions, discount):
    '''
    Sample termination time from a geometric distribution with parameter (1 - discount) and the total undiscounted cost independently. Compute the empirical mean.

    Ref:
        C. Derman, Finite State Markoc Decision Processes, Academic Press, New York, 1970.
        B. Fox and P. Glynn, "Simulating discounted cost", Management Sci. 35, 1297-1315 (1989).
    '''
    n_steps = len(transitions)
    visit = np.zeros(n_states)
    empirical_mean = np.zeros(n_states)
    for t, (state, reward) in enumerate(transitions):
        # Draw a termination time from the geometric distribution with p = 1 - discount (terminate w.p. p)
        ter = np.random.geometric(1 - discount)
        # Return early if there are not enough steps left
        if n_steps < t + ter:
            return empirical_mean
        ss, rr = zip(*transitions[t:t + ter])
        # Compute the total undiscounted reward
        undiscounted = np.sum(rr)
        # Update the empirical mean
        empirical_mean[state] = (visit[state] * empirical_mean[state] + undiscounted) / (visit[state] + 1)
        # Update visitation counts
        visit[state] += 1
    return empirical_mean


def geometric_single(state, transitions, discount, seed=None):
    random = np.random.RandomState(seed)
    ss, rr = zip(*transitions)
    t0 = ss.index(state)
    tl = len(transitions)
    n_fragments = 0
    undiscounted_return = 0
    v_hats = []
    while t0 < tl:
        fragment_len = random.geometric(1 - discount)
        if tl - t0 < fragment_len:
            # Not enough transitions, terminate
            break
        # New sample fragment
        undiscounted_fragment_return = np.sum(rr[t0:t0 + fragment_len])
        # Update estimates
        undiscounted_return = n_fragments / (n_fragments + 1) * undiscounted_return + 1 / (n_fragments + 1) * undiscounted_fragment_return
        n_fragments += 1
        v_hats.append(undiscounted_return)
        # Find the next visit
        try:
            t0 = t0 + fragment_len + ss[t0 + fragment_len:].index(state)
        except ValueError:
            # No more visits, terminate
            break
    v_hat = undiscounted_return
    return v_hat, v_hats


def td(n_states, transitions, alpha_func, discount):
    v_hat = np.zeros(n_states)
    visit = np.zeros(n_states)
    for t, (state, reward) in enumerate(transitions[:-1]):
        next_state, _ = transitions[t + 1]
        alpha = alpha_func(visit[state])
        v_hat[state] = alpha * (reward + discount * v_hat[next_state]) + (1 - alpha) * v_hat[state]
        visit[state] += 1
    return v_hat


def td_k(k, n_states, transitions, discount):
    # Initial estimates for state values
    v_hat = np.zeros(n_states)
    visit = np.zeros(n_states)
    gammas = discount ** np.arange(k)
    ss, rr = zip(*transitions)
    for t, state in enumerate(ss[:-k]):
        next_state = ss[t + k]
        alpha = 1 / (visit[state] + 1)
        # Update estimate
        partial_return = np.sum(gammas * rr[t:t+k])
        v_hat[state] = alpha * (partial_return + (discount ** k) * v_hat[next_state]) + (1 - alpha) * v_hat[state]
        visit[state] += 1
    return v_hat


def loop(n_states, transitions, discount):
    # Find the recurrences of the same state
    reward_return = defaultdict(lambda: [])
    # Saving the partial return and time of the last visit to x
    last_visit = {}
    for t, (x, r) in enumerate(transitions[:-1]):
        if x in last_visit:
            partial_return, last_t = last_visit[x]
            reward_return[x].append((partial_return, t - last_t))
        last_visit[x] = [0, t]
        for xx in last_visit:
            last_t = last_visit[xx][1]
            last_visit[xx][0] += discount ** (t - last_t) * r
    # Compute the estimates
    v_pi = np.zeros(n_states)
    for xx in reward_return:
        prs, dts = zip(*reward_return[xx])
        v_pi[xx] = np.mean(prs) / np.mean(1 - np.power(discount, dts))
    # TODO states that did not recurr have value 0
    # TODO states that has a running sequence (there are n_states-1 many of them)
    # TODO states that did not appear
    return v_pi


def loop_single(state, transitions, discount):
    # Delayed renewal process
    last_visited_at = None
    running_discounted_return = 0
    visits = 0
    renewal_discount = 0
    renewal_discounted_return = 0
    v_hats = []
    for s, r in transitions:
        if s == state:
            if last_visited_at is not None:
                # End of a loop
                # Update estimates
                n = visits + 1
                renewal_discount = visits / n * renewal_discount + 1 / n * (discount ** last_visited_at)
                renewal_discounted_return = visits / n * renewal_discounted_return + 1 / n * running_discounted_return
                visits += 1
                v_hats.append(renewal_discounted_return / (1 - renewal_discount))
            # Reset
            last_visited_at = 0
            running_discounted_return = 0
        # Update the running partial return
        if last_visited_at is not None:
            running_discounted_return += (discount ** last_visited_at) * r
            # Increment the clock
            last_visited_at += 1
    # Plug in the renewal reward identity
    v_hat = renewal_discounted_return / (1 - renewal_discount)
    return v_hat, v_hats


def model_based(n_states, transitions, discount, state=None):
    # O(S^2) space
    if state is None:
        p_hat = np.zeros((n_states, n_states))
        r_hat = np.zeros(n_states)
        visit = np.zeros(n_states)
    else:
        p_hat, r_hat, visit = state
    for t, (x, r) in enumerate(transitions[:-1]):
        y, _ = transitions[t + 1]
        r_hat[x] += r
        p_hat[x, y] += 1
        visit[x] += 1
    if np.any(0 == visit):
        return np.zeros(n_states), (p_hat, r_hat, visit)
    p_hat /= p_hat.sum(1).reshape((-1, 1))
    r_hat /= visit
    # Plug-in estimator
    # O(n^3) for matrix inversion
    v_hat = exact(p_hat, r_hat, discount)
    return v_hat, (p_hat, r_hat, visit)


if __name__ == '__main__':
    n_states = 10
    discount = 0.99
    n_steps = 1000
    p = np.zeros((n_states, n_states))
    for i in range(n_states):
        # p[i] = np.random.dirichlet([1] * n_states)
        # alpha = np.random.randint(1, 100, n_states)
        # print(alpha)
        # p[i] = np.random.dirichlet(alpha)
        p[i] = np.random.dirichlet([100] * i + [1] * (n_states - i))
    print(p)
    # r = np.random.rand(n_states)
    r = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
    print(r)
    mrp = MarkovRewardProcess(n_states, p, r, initial_state=0)

    v = exact(p, r, discount)
    print(v)

    sampler = mrp.sample_transition()
    samples = [next(sampler) for _ in range(n_steps)]

    v_hat = geometric(n_states, samples, discount)
    print(v_hat)
    print('geometric', np.max(np.abs(v - v_hat)))

    v_hat = truncate(n_states, samples, 100, discount)
    print(v_hat)
    print('truncate', np.max(np.abs(v - v_hat)))

    # alpha_func = lambda n: 1 / np.sqrt(1 + n)
    alpha_func = lambda n: 1 / (1 + n)
    # alpha_func = lambda n: 0.1
    v_hat = td(n_states, samples, alpha_func, discount)
    print(v_hat)
    print('td', np.max(np.abs(v - v_hat)))

    v_hat = loop(n_states, samples, discount)
    print(v_hat)
    print('loop', np.max(np.abs(v - v_hat)))

    v_hat = model_based(n_states, samples, discount)
    print(v_hat)
    print('model_based', np.max(np.abs(v - v_hat)))
