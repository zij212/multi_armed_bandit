import time
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """
    create k armed bernoulli bandit with given or randomly assigned probability p
    """
    def __init__(self, k, p=None):
        assert p is None or len(p) == k
        self.k = k

        np.random.seed(int(time.time()))
        if p is None:
            self.p = [np.random.random() for _ in range(self.k)]
        else:
            self.p = p

        # self.max_p is used later to calculate regret
        self.max_p = max(self.p)

    def get_reward(self, i):
        """
        play the i_th action and return the result
        :param i: index of an action
        :return: reward 1 or 0 with the probability self.p[i], (1- self.p[i])
        """
        return int(np.random.random() < self.p[i])


class ExploreExploitAlgorithm:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(bandit.k, dtype=int)
        self.actions = []
        self.regret = 0
        self.regrets = [0]

    def step(self):
        raise NotImplementedError

    @property
    def estimates(self):
        raise NotImplementedError

    def update_regret(self, i):
        self.regret += self.bandit.max_p - self.bandit.p[i]
        self.regrets.append(self.regret)

    def run(self, num_steps):
        for s in range(num_steps):
            i = self.step()
            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(ExploreExploitAlgorithm):
    def __init__(self, bandit, epsilon):
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0 <= epsilon <= 1

        self.name = "Epsilon-Greedy"

        self.epsilon = epsilon
        self.p_hats = np.ones(bandit.k, dtype=np.float)

    @property
    def estimates(self):
        return self.p_hats

    def step(self):
        """
        :return: index of the best action (selected by epsilon-greedy algorithm) to play
        """

        if np.random.random() < self.epsilon:
            i = np.random.randint(0, self.bandit.k)
        else:
            i = np.argmax(self.p_hats)

        # play to collect reward
        r = self.bandit.get_reward(i)

        # update self.p_hats accordingly
        self.p_hats[i] += (r - self.p_hats[i]) / (self.counts[i] + 1)

        return i


class ThompsonSampling(ExploreExploitAlgorithm):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)

        self.name = "Thompson Sampling"

        self.a = np.ones(bandit.k, dtype=int)
        self.b = np.ones(bandit.k, dtype=int)

    @property
    def estimates(self):
        return self.a / (self.a + self.b)

    def step(self):
        # sample from all bandits and select bandit i that argmax_i(beta(a_i, b_i))
        samples = np.random.beta(self.a, self.b)
        i = np.argmax(samples)

        # update a_i, b_i using the reward
        r = self.bandit.get_reward(i)
        self.a[i] += r
        self.b[i] += 1 - r

        return i


class UCB1(ExploreExploitAlgorithm):
    def __init__(self, bandit):
        super(UCB1, self).__init__(bandit)

        self.name = "UCB1"
        self.p_hats = np.ones(self.bandit.k, dtype="float")
        self.num_games_played = 0

    @property
    def estimates(self):
        return self.p_hats

    def step(self):
        # sample from all bandits and select $i^*=argmax_i(\hat{\mu}_i+\sqrt{\frac{2lnN}{N_i}})$
        # where N is total number of games played, and N_i is the number of times bandit i us played
        self.num_games_played += 1  # np.log(1)
        samples = self.p_hats + np.sqrt(2 * np.log(self.num_games_played) / (1 + np.array(self.counts)))
        i = np.argmax(samples)

        # get reward r_i and update the estimate for p_i
        r = self.bandit.get_reward(i)
        self.p_hats[i] += (r - self.p_hats[i]) / (self.counts[i] + 1)

        return i
    

def plot(algorithms, k, n):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, alg in enumerate(algorithms):
        ax[0].plot(range(len(alg.regrets)), alg.regrets, label=alg.name)

    ax[0].set_xlabel("time step")
    ax[0].set_ylabel("cumulative regret")

    sorted_bandit_index = sorted(range(algorithms[0].bandit.k), key=lambda x: algorithms[0].bandit.p[x])

    # estimated probability
    for alg in algorithms:
        ax[1].scatter(range(alg.bandit.k), [alg.estimates[i] for i in sorted_bandit_index], marker='x', label=alg.name)
    # true probability
    ax[1].scatter(range(alg.bandit.k), [algorithms[0].bandit.p[i] for i in sorted_bandit_index],
                  label="True probability", marker='x', color='black')
    ax[1].set_xlabel(r"action sorted by true $\theta$")
    ax[1].set_ylabel(r"$\hat\theta}$", rotation=0)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    ax[1].legend(bbox_to_anchor=(1.5, -0.25), ncol=5)

    # num of times an armed is played over total times played
    for alg in algorithms:
        ax[2].plot(range(alg.bandit.k), [(alg.counts/sum(alg.counts))[i] for i in sorted_bandit_index], label=alg.name, drawstyle='steps')
    ax[2].set_xlabel(r"action sorted by true $\theta$")
    ax[2].set_ylabel(f"fraction played over {n} games")

    plt.savefig(f"./result_for_{k}_armed_bandit_in_{n}_steps.png")


def run_experiment(k: int, n: int):
    """

    :param k: number of machines
    :param n: number of time steps
    :return:
    """
    b = BernoulliBandit(k)
    algs = [
        EpsilonGreedy(b, 0.01),
        UCB1(b),
        ThompsonSampling(b),
    ]
    for alg in algs:
        alg.run(n)
    plot(algs, k, n)


if __name__ == '__main__':
    run_experiment(10, 10000)

