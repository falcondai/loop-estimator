import numpy as np


class MarkovRewardProcess:
    def __init__(self, n_states, p, r, initial_state, r_max=1, random_reward=False, seed=None):
        '''
        Args:
            p : (n_states, n_states)-array.
                Transition matrix with the common row-column convention. p[i, j] := transition probability from state i to state j.
        '''
        self.random = np.random.RandomState(seed=seed)
        self.n_states = n_states
        assert initial_state < n_states
        self.state = initial_state
        assert p.shape == (n_states, n_states)
        self.p = p
        assert r.shape == (n_states, )
        assert (r <= r_max).all(), 'The mean rewards should be no greater than r_max of %r' % r_max
        self.r = r
        self.r_max = r_max
        # Draw from a Bernoulli distribution {0, r_max} with parameter r[state] / r_max
        self.random_reward = random_reward

    def step(self):
        if self.random_reward:
            reward = self.r_max if self.random.rand() < self.r[self.state] / self.r_max else 0
        else:
            reward = self.r[self.state]
        self.state = self.random.choice(self.n_states, p=self.p[self.state])
        return self.state, reward

    def sample_transition(self):
        while True:
            yield self.step()


class MarkovDecisionProcess:
    def __init__(self, n_states, n_actions, p, r, initial_state, r_max=1, random_reward=False, seed=None):
        '''
        Args:
            p : (n_actions, n_states, n_states)-array.
                Transition matrix with the common row-column convention. p[k, i, j] := transition probability from state i to state j under action k.
        '''
        self.random = np.random.RandomState(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        assert initial_state < n_states
        self.state = initial_state
        assert p.shape == (n_actions, n_states, n_states)
        self.p = p
        assert r.shape == (n_actions, n_states, )
        assert (r <= r_max).all(), 'The mean rewards should be no greater than r_max of %r' % r_max
        self.r = r
        self.r_max = r_max
        # Draw from a Bernoulli distribution {0, r_max} with parameter r[action, state] / r_max
        self.random_reward = random_reward

        def step(self, action):
            if self.random_reward:
                reward = self.r_max if self.random.rand() < self.r[action, self.state] / self.r_max else 0
            else:
                reward = self.r[action, self.state]
            # Next state
            self.state = self.random.choice(self.n_states, p=self.p[action, self.state])
            return action, self.state, reward

        def sample_transition(self):
            while True:
                yield self.step()


class MdpPi(MarkovRewardProcess):
    def __init__(self, mdp, pi):
        n_states = mdp.n_states
        n_actions = mdp.n_actions
        assert 0 <= max(pi) < n_actions
        assert len(pi) == n_states

        p_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)
        for state in range(n_states):
            action = pi[state]
            p_pi[state] = mdp.p[action, state]
            r_pi[state] = mdp.r[action, state]
        super(MdpPi, self).__init__(n_states, p_pi, r_pi, mdp.state, mdp.r_max, mdp.random_reward, None)


class RiverSwim(MarkovDecisionProcess):
    def __init__(self, n_states=6, initial_state=0, random_reward=False, seed=None):
        # Ref: [SL08] An analysis of model-based Interval Estimation for Markov Decision Processes.
        n_actions = 2
        # action 0 : swim left
        # action 1 : swim right
        r_max = 10 ** 4 * 0.3
        p = np.zeros((n_actions, n_states, n_states))
        r = np.zeros((n_actions, n_states))
        # Leftmost state is less rewarding
        r[0, 0] = 5
        r[1, -1] = r_max
        for st in range(n_states):
            # Moving left is reliable
            p[0, st, max(0, st - 1)] = 1
            # Moving right sometimes succeeds
            p[1, st, min(n_states - 1, st + 1)] += 0.3
            p[1, st, st] += 0.6
            p[1, st, max(0, st - 1)] += 0.1
        # Rightmost state is special
        p[1, -1, -1] = 0.3
        p[1, -1, -2] = 0.7

        super(RiverSwim, self).__init__(n_states, n_actions, p, r, initial_state, r_max, random_reward, seed)


if __name__ == '__main__':
    p = np.zeros((2, 2))
    p[0, 0] = 0.9
    p[0, 1] = 0.1
    p[1, 1] = 1
    r = np.zeros(2)
    r[0] = 0.9
    r[1] = 0.1
    mrp = MarkovRewardProcess(2, p, r, 0, r_max=0.9, random_reward=True)
    for t, (st, rew) in enumerate(mrp.sample_transition()):
        print(t, st, rew)
        if 20 < t:
            break

    mdp = RiverSwim()
    pi = [1] * 6
    mdp_pi = MdpPi(mdp, pi)
    print(mdp_pi.p)
    for t, (st, rew) in enumerate(mdp_pi.sample_transition()):
        print(t, st, rew)
        if 20 < t:
            break
