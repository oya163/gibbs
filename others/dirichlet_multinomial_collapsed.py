# -*- coding: utf-8 -*-

"""
    Dirichlet Multinomial Distribution
    Collapsed Sampling
    Best result:
    python dirichlet_multinomial_collapsed.py -i 14 -e 1000 -k 2 -s 2 -f 1
    python dirichlet_multinomial_collapsed.py -i 6 -e 2000 -k 2 -s 2 -f 1
    python dirichlet_multinomial_collapsed.py -i 16 -e 1000 -k 2 -s 2 -f 2
    python dirichlet_multinomial_collapsed.py -i 15 -e 500 -k 2 -s 2 -f 3
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import multinomial
import random
import math
import argparse

np.random.seed(200)

# To display ndarray upto 3rd precision
np.set_printoptions(precision=3)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Mixture Model")
    parser.add_argument("-k", "--cluster", dest="cluster", type=int,
                        default=2, metavar="INT", help="Cluster size [default:3]")
    parser.add_argument("-i", "--iteration", dest="iteration", type=int,
                        default=50, metavar="INT", help="Iterations [default:100]")
    parser.add_argument("-e", "--experiments", dest="experiments", type=int,
                        default=500, metavar="INT", help="Total Experiments [default:500]")
    parser.add_argument("-f", "--flips", dest="flips", type=int,
                        default=2, metavar="INT", help="Total flips per experiment [default:10]")
    parser.add_argument("-s", "--sides", dest="sides", type=int,
                        default=3, metavar="INT", help="Total sides of a dice [default:6]")
    args = parser.parse_args()
    return args


class MultinomialMixtureModel:
    def __init__(self, K, S, total_iteration, total_experimentation, total_flip_per_experimentation):
        # number of cluster
        self.K = K
        self.S = S

        # Generate random alphas in the range [0.0, 1.0)
        # Keep alphas and beta low
        self.alpha = [np.random.random() for k in range(self.K)]
        self.beta = [np.random.randint(1,5) for k in range(self.S)]

        # Sample initial theta for each cluster from
        # dirichlet distribution so that it sum to 1
        # Produces theta based on number of sides
        # Keep dirichlet_init low, init_theta produced is not equally distributed among sides
        # Produces good result
        self.dirichlet_init = [np.random.random() for x in range(self.S)]
        self.init_theta = np.random.dirichlet(alpha=self.dirichlet_init, size=self.K)

        self.total_flip_per_experimentation = total_flip_per_experimentation
        self.total_experimentation = total_experimentation

        # Random dirichlet distribution for mixture proportion
        self.actual_mixture = np.random.dirichlet(alpha=[np.random.randint(1, 100) for x in range(self.K)], size=1)[0]

        # Sum of alphas
        self.A = sum(self.alpha)
        self.B = sum(self.beta)

        # Number of iteration
        self.total_iteration = total_iteration

        # Total performance
        self.total_performance = []

    # Data generation in each experiment based on given probability
    def experiments(self, prob):
        res = []
        for i in range(self.total_flip_per_experimentation):
            choice = np.random.choice([x for x in range(self.S)], 1, p=prob, replace=False)[0]
            res.append(choice)
        return res

    # Generate data
    def generate_data(self):
        data = []

        # Generate data
        for i in range(self.total_experimentation):
            # Get a experiment type based on dirichlet distribution
            experiment_type = np.random.choice([x for x in range(self.K)], 1, p=self.actual_mixture)[0]

            # Get experiment results based on initial theta of chosen experiment type
            experiment_result = self.experiments(self.init_theta[experiment_type])

            # Append each experiment result into data
            data.append(experiment_result)

        # display actual parameters
        self.display_actual()

        return data

    # Multinomial distribution
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html
    def g0(self, data, p):
        # Get counter dictionary of given data
        count = dict(sorted(Counter(data).items()))
        x = []

        # Getting the values of count else put 0
        for k in range(self.S):
            if k in count:
                x.append(count[k])
            else:
                x.append(0)

        # Numerator part
        # numerator = math.factorial(
        #     self.total_flip_per_experimentation) * np.prod(
        #     [math.pow(prob, v) for prob, (k, v) in zip(p, count.items())])

        # First get list of factorials of all values, then multiply all together
        # Denominator part
        # denominator = np.prod([math.factorial(v) for k, v in count.items()])

        # result = numerator/denominator
        result = multinomial.pmf(np.array(x), n=self.total_flip_per_experimentation, p=p)
        return result

    # Uniform random distribution of initial cluster assignment
    def initialize_cluster(self):
        init_cluster = []
        for i in range(self.total_experimentation):
            experiment_type = np.random.choice([x for x in range(self.K)], 1, p=[1 / self.K for x in range(self.K)])[0]
            init_cluster.append(experiment_type)

        return init_cluster

    # Remove data of given index from current cluster
    def remove_current_data(self, index, input_data, cluster_id):
        new_cluster = {}
        # Initialize the fix number of cluster first
        for k in range(self.K):
            new_cluster[k] = []

        # Accumulate all the data except from the given index
        for idx, d in enumerate(input_data):
            # Skip the given index
            if idx != index:
                new_cluster[cluster_id[idx]].extend(d)
        return new_cluster

    def _posterior_predictive(self, within_index, except_index):
        alpha_dash = [0.0 + self.beta[s] for s in range(self.S)]
        y_j = [1] * self.S
        # Just take the count of such elements which are present in y_j
        for within_key, within_val in within_index.items():
            y_j.append(within_val)
            alpha_dash[within_key] = except_index[within_key] + self.beta[within_key]

        first_numer = math.factorial(self.total_flip_per_experimentation)
        first_denom = np.prod([math.factorial(x) for x in y_j])
        first_term = first_numer/first_denom

        second_numer = math.factorial(sum(alpha_dash) - 1)
        second_denom = 1
        for x in alpha_dash:
            second_denom *= math.factorial(x-1)
        second_term = second_numer/second_denom

        y_plus_alpha_dash = 1
        for a, b in zip(y_j, alpha_dash):
            y_plus_alpha_dash *= math.factorial(a+b-1)
        third_numer = y_plus_alpha_dash
        third_denom = math.factorial(sum(alpha_dash) + self.total_flip_per_experimentation - 1)
        third_term = third_numer/third_denom

        result = second_term * third_term
        return result

    def _multinomial_predictive(self, within_cluster):
        C = sum(within_cluster.values())
        c_i = within_cluster.values()
        A = sum(self.beta)

        first_numerator = math.factorial(C)
        first_denom = 1
        for each in c_i:
            first_denom *= math.factorial(each)
        first_term = first_numerator/first_denom

        second_numerator = math.factorial(A - 1)
        second_denom = 1
        for each in self.beta:
            second_denom *= math.factorial(each)
        second_term = second_numerator/second_denom

        third_numerator = 1
        for c, a in zip(c_i, self.beta):
            third_numerator *= math.factorial(c+a-1)
        third_denominator = math.factorial(C + A - 1)
        third_term = third_numerator/third_denominator

        return first_term * second_term * third_term


    # Calculate each cluster assignment probability
    def calculate_cluster_assignment_probability(self, index, X, init_cluster):
        N = self.total_experimentation

        # Compute cluster counts + alphas
        count_c_k = Counter(init_cluster)
        c_k = []
        for i, a in zip(range(self.K), self.alpha):
            c_k.append(count_c_k[i] + a)

        xi_side = X[index] # [1, 1]

        within_experiment = Counter(xi_side)

        # Remove current data
        cluster = self.remove_current_data(index, X, init_cluster)

        n_k = {}
        posterior_predictive = []
        for k, v in cluster.items():
            within_cluster = Counter(v)
            # posterior_predictive.append(self._posterior_predictive(within_experiment, within_cluster))
            posterior_predictive.append(self._multinomial_predictive(within_cluster))

        # Compute p_k (x_i | theta_k) for each cluster
        final_prob = []
        for k in range(self.K):
            final_prob.append((c_k[k] / (N + self.A - 1)) * posterior_predictive[k])

        return posterior_predictive, final_prob

    # Gibbs sampling on Dirichlet-Multinomial distribution
    def fit(self, data, init_cluster):
        total_performance = []
        best_performance = 0.0
        best_cluster = []
        best_theta = []

        for num_iteration in range(self.total_iteration):
            # print("Num iteration", num_iteration)
            performance = []
            for i in range(len(data)):
                # print("Data i: ", i)
                # Compute dirichlet-multinomial distribution
                theta, final_prob = self.calculate_cluster_assignment_probability(i, data, init_cluster)

                # Normalize the probability
                norm_prob = final_prob / np.sum(final_prob)

                # Update the cluster assignment
                init_cluster[i] = np.random.choice([x for x in range(self.K)], 1, p=norm_prob)[0]

                # Computer log-likelihood
                performance.append(np.log(np.sum(final_prob)))

            # curr_performance = np.sum(performance)
            # if curr_performance < best_performance:
            #     print(curr_performance)
            #     best_theta = theta
            #     best_cluster = init_cluster
            #     best_performance = curr_performance

            total_performance.append(np.sum(performance))

        print("Minimum log-likelihood at ", np.argmax(total_performance))

        return init_cluster, theta, total_performance

    # Display actual parameters
    def display_actual(self):
        print("************ Actual Parameters ************")
        for k in range(self.K):
            print("Cluster Id: {}, Mixture: {:.3f}".format(k, self.actual_mixture[k]))

    # Display estimated parameters
    def display_estimated(self, init_cluster):
        print("************ Estimated Parameters ************")
        cluster_type = Counter(init_cluster)

        for k in range(self.K):
            print(
                "Cluster Id: {}, Mixture: {:.3f}".format(k, cluster_type[k] / len(init_cluster)))

    # Display log likelihood plot
    def display_plot(self, total_performance):
        # Display plot
        plt.figure(figsize=(10, 5))
        plt.plot(total_performance)
        plt.title("Log-Likelihood")
        plt.xlabel("Number of Iteration")
        plt.ylabel("log p(x_i,z_i|X_-i, Z_-i, theta)")
        plt.xticks([i for i in range(self.total_iteration)])
        plt.show()


def main():
    args = parse_args()

    # Get arguments
    K = args.cluster
    S = args.sides
    total_iteration = args.iteration
    total_experimentation = args.experiments
    total_flip_per_experimentation = args.flips

    # define model
    model = MultinomialMixtureModel(K, S, total_iteration, total_experimentation, total_flip_per_experimentation)

    # generate data
    data = model.generate_data()

    # uniform cluster assignment
    init_cluster = model.initialize_cluster()

    # gibbs sampling
    init_cluster, theta, total_performance = model.fit(data, init_cluster)

    # display estimated parameters
    model.display_estimated(init_cluster)

    # display performance plot
    model.display_plot(total_performance)


if __name__ == "__main__":
    main()
