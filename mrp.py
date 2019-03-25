import numpy as np


class MarkovRewardProcess:
    def __init__(self, n_states, p, r, initial_state, seed=None):
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
        self.r = r

    def step(self):
        reward = self.r[self.state]
        self.state = self.random.choice(self.n_states, p=self.p[self.state])
        return reward, self.state

    def sample_transition(self):
        while True:
            yield self.state, self.r[self.state]
            self.step()


if __name__ == '__main__':
    p = np.zeros((2, 2))
    p[0, 0] = 0.9
    p[0, 1] = 0.1
    p[1, 1] = 1
    r = np.zeros(2)
    r[0] = 1
    r[1] = 0
    mrp = MarkovRewardProcess(2, p, r, 0)
    for t, (st, rew) in enumerate(mrp.sample_transition()):
        print(t, st, rew)
        if 20 < t:
            break
    # for t in range(20):
    #     print(t, mrp.step())
