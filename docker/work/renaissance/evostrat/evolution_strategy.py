from __future__ import print_function
import numpy as np
import multiprocessing as mp
import pickle
import time
np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, savepath, population_size=50, sigma=0.1,
                 learning_rate=0.03, decay=0.999, num_threads=1):

        self.weights = weights
        self.get_reward = get_reward_func
        self.save_path = savepath
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        """Get the population for the genetic algorithm."""
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        """
        Get rewards for a given population using parallel processing if a pool is provided, otherwise use a sequential approach.
        
        Args:
            pool: The parallel processing pool, if any.
            population: The population for which rewards need to be calculated.
        
        Returns:
            numpy array: An array of rewards for the given population.
        """
        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            start = time.time()
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        print(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        """
        Update the weights of the agent based on the rewards and the population.

        Parameters:
            rewards (array): The rewards obtained from the population.
            population (array-like): The population of solutions.

        Returns:
            None
        """
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + update_factor * np.dot(layer_population.T, rewards).T
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=1):
        """
        Run the genetic algorithm for a specified number of iterations.

        Parameters:
            iterations (int): The number of iterations to run the genetic algorithm.
            print_step (int, optional): The frequency with which to print progress. Defaults to 1.

        Returns:
            numpy.ndarray: An array of rewards obtained at each iteration.
        """
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        print('starting')
        start = time.time()
        all_rewards = []

        for iteration in range(iterations):

            population = self._get_population()

            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)

            this_reward = self.get_reward(self.weights)

            #save weights
            with open(f'{self.save_path}/weights_{iteration}.pkl', 'wb') as f:
                    pickle.dump(self.weights, f)

            if (iteration + 1) % print_step == 0:
                print('********iter %d. reward: %f********' % (iteration + 1, this_reward ))
            all_rewards.append(this_reward)
            this_end = time.time()
            print(f'Time elapsed: {round((this_end-start)/60, 3)} minutes')
        if pool is not None:
            pool.close()
            pool.join()

        return np.array(all_rewards)
