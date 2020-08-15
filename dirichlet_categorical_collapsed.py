# -*- coding: utf-8 -*-

"""
    Dirichlet Categorical Distribution
    Collapsed Sampling
    Best result:
    python dirichlet_categorical_collapsed.py -e 100 -k 2 -s 2 -f 1 -i 2
    python dirichlet_categorical_collapsed.py -i 5 -e 1000 -k 2 -s 2 -f 1
    python dirichlet_categorical_collapsed.py -i 6 -e 2000 -k 2 -s 2 -f 1
    python dirichlet_categorical_collapsed.py -i 18 -e 100 -k 3 -s 2 -f 1
    python dirichlet_categorical_collapsed.py -e 300 -k 4 -s 2 -f 1 -i 4
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import multinomial
import random
import math
import argparse

np.random.seed(163)

# To display ndarray upto 3rd precision
np.set_printoptions(precision=3)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Mixture Model")
    parser.add_argument("-k", "--cluster", dest="cluster", type=int,
                        default=3, metavar="INT", help="Cluster size [default:3]")
    parser.add_argument("-i", "--iteration", dest="iteration", type=int,
                        default=100, metavar="INT", help="Iterations [default:100]")
    parser.add_argument("-e", "--experiments", dest="experiments", type=int,
                        default=100, metavar="INT", help="Total Experiments [default:500]")
    parser.add_argument("-f", "--flips", dest="flips", type=int,
                        default=1, metavar="INT", help="Total flips per experiment [default:10]")
    parser.add_argument("-s", "--sides", dest="sides", type=int,
                        default=6, metavar="INT", help="Total sides of a dice [default:6]")
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

    # Calculate each cluster assignment probability
    def calculate_cluster_assignment_probability(self, index, X, init_cluster):
        N = self.total_experimentation

        # Compute cluster counts + alphas
        count_c_k = Counter(init_cluster)
        c_k = []
        for i, a in zip(range(self.K), self.alpha):
            c_k.append(count_c_k[i])

        xi_side = X[index][0]

        # Remove current data
        cluster = self.remove_current_data(index, X, init_cluster)

        posterior_predictive = []
        for k, v in cluster.items():
            # curr_theta = [0.0+self.alpha[k] for k in range(self.K)]
            # c_v = dict(sorted(Counter(v).items()))
            c_v = Counter(v)
            try:
                c_k_xi = c_v[xi_side]
            except KeyError:
                c_k_xi = 0
            posterior_predictive.append((c_k_xi + self.beta[xi_side]) / (len(v) + self.B - 1))

            # # In case, there are no values in given cluster
            # curr_theta = [0.0+self.beta[s] for s in range(self.S)]
            # sides = dict(sorted(Counter(v).items()))
            # for side_key, side_val in sides.items():
            #     # curr_theta[c] = val / len(v)        # MLE
            #     curr_theta[side_key] = side_val + self.beta[side_key]
            # # Sample theta from dirichlet
            # theta.append(np.random.dirichlet(np.array(curr_theta), size=1)[0])

        # Compute p_k (x_i | theta_k) for each cluster
        final_prob = []
        for k in range(self.K):
            final_prob.append(((c_k[k] + (self.A / self.K)) / (N + self.A - 1)) * posterior_predictive[k])

        return posterior_predictive, final_prob

    # Gibbs sampling on Dirichlet-Multinomial distribution
    def fit(self, data, init_cluster):
        total_performance = []
        best_performance = 0.0
        best_cluster = []
        best_theta = []

        for num_iteration in range(self.total_iteration):
            performance = []
            for i in range(len(data)):
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
