# -*- coding: utf-8 -*-
"""
    Description:
    Infinite Gaussian Mixture Model
    Based on Chinese Restaurant Process
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import invgamma

import argparse

np.random.seed(163)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Mixture Model")
    parser.add_argument("-k", "--cluster", dest="cluster", type=int,
                        default=3, metavar="INT", help="Cluster size [default:3]")
    parser.add_argument("-i", "--iteration", dest="iteration", type=int,
                        default=50, metavar="INT", help="Iterations [default:50]")
    parser.add_argument("-a", "--alpha", dest="alpha", type=float,
                        default=0.9, metavar="FLOAT", help="Alpha")
    parser.add_argument("-t", "--total_points", dest="max_range", type=int,
                        default=500, metavar="INT", help="Total Points [default:500]")
    parser.add_argument("-p", "--prior", dest="prior", type=str, default='mle',
                        choices=['mle', 'ngi'], metavar="STR", help="Prior Selection [default:mle]")
    args = parser.parse_args()
    return args


# Normal base function
def normal_base(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- ((x - mu) ** 2) / (2 * (sigma ** 2)))


def inverse_gamma(x, alpha, beta):
    first = math.pow(beta, alpha)
    second = math.pow(1 / x, (alpha + 1))
    third = math.exp(-(beta / x))
    numerator = first * second * third
    denominator = math.factorial(alpha - 1)
    return numerator / denominator


class MixtureModel:
    def __init__(self, K, total_iteration, A, total_points, prior):
        # Number of cluster
        self.K = K
        self.total_points = total_points

        self.points = []
        self.actual_mu = []
        self.actual_sigma = []
        self.actual_mixture = []

        self.A = A
        self.total_iteration = total_iteration
        self.prior = prior

        self.mu_0 = 0
        self.sigma_0 = 5
        self.alpha_0 = 2
        self.beta_0 = 7

    def generate_data(self):
        data = []
        mu_list = np.linspace(0, 20, self.K, dtype=int)
        sigma_list = np.arange(1, self.K + 1)

        # Random proportion generation from dirichlet
        alpha = [np.random.randint(10, 50) for x in range(self.K)]
        proportion = np.random.dirichlet(alpha=alpha, size=1)[0]

        # Generate data
        id_list = []
        for i in range(self.total_points):
            id = np.random.choice(self.K, 1, p=proportion)[0]
            id_list.append(id)
            data.extend(np.random.normal(mu_list[id], sigma_list[id], 1))

        # Just for display and analysis purpose
        id_counter = Counter(id_list)
        for k in range(self.K):
            mu = mu_list[k]
            sigma = sigma_list[k]
            self.actual_mu.append(mu)
            self.actual_sigma.append(sigma)
            self.actual_mixture.append(id_counter[k] / self.total_points)

        self.display_actual()

        return data

    # Random assignment by generating random number
    # between 0 and given number of cluster
    def initialize_cluster(self, data):
        init_data = []
        for each in data:
            init_data.append(np.random.randint(0, self.K))
        return init_data

    # Remove given data from cluster
    def remove_current_data(self, index, input_data, orig_data):
        new_cluster = {}

        for idx, d in enumerate(input_data):
            # Skip the given index
            if idx != index:
                # append data to created cluster id
                if orig_data[idx] in new_cluster:
                    new_cluster[orig_data[idx]].append(d)
                # create new cluster id
                else:
                    new_cluster[orig_data[idx]] = [d]
        return new_cluster

    def fit(self, data, init_data):
        N = len(data)
        total_performance = []
        for itr in range(self.total_iteration):
            performance = []
            for i, x in enumerate(data):
                # Remove data point
                cluster = self.remove_current_data(i, data, init_data)

                # Compress the cluster
                # to prevent from ever increasing cluster id
                keys = sorted(cluster.keys())
                for j in range(0, len(keys)):
                    cluster[j] = cluster.pop(keys[j])
                # print("Dictionary keys:", cluster.keys())

                # Parameters
                mu = []
                sigma = []
                cluster_prob = []
                final_prob = []

                # Estimate Gaussian parameters of each cluster
                for k, v in sorted(cluster.items()):
                    try:
                        n = len(cluster[k])
                        x_bar = np.mean(cluster[k])

                        # Calculate MLE
                        if self.prior == "mle":
                            mu.append(x_bar)
                            sigma_mle = 1.0 if n == 1 else np.sqrt(np.sum((cluster[k] - mu[-1]) ** 2) / n)
                            sigma.append(sigma_mle)
                        # Normal Inverse Gamma prior
                        elif self.prior == "ngi":
                            # For unknown variance
                            alpha = self.alpha_0 + n / 2
                            beta = self.beta_0 + ((np.sum([((x_i - x_bar) ** 2) for x_i in cluster[k]])) / 2)
                            variance = invgamma.rvs(alpha, scale=beta)
                            post_sigma = np.sqrt(variance)

                            # With known variance
                            S = 1 / ((1 / self.sigma_0) + (n / variance))
                            mu_1 = S * ((self.mu_0 / self.sigma_0) + (np.sum(cluster[k]) / variance))
                            sigma_1 = np.sqrt(S)
                            post_mu = np.random.normal(mu_1, sigma_1)

                            mu.append(post_mu)
                            sigma.append(post_sigma)

                        # Get base distribution
                        cluster_prob.append(normal_base(x, mu[-1], sigma[-1]))
                        # cluster_prob.append(norm.pdf(x, loc=mu[k], scale=sigma[k]))

                        # Count of data points in each cluster
                        likelihood = n / (N + self.A - 1)
                        final_prob.append(likelihood * cluster_prob[-1])

                    except KeyError as e:
                        continue

                # Probability of joining new cluster
                final_prob.append((self.A / (N + self.A - 1)) * normal_base(x, mu=1, sigma=2))

                # Normalize the probability
                norm_prob = final_prob / np.sum(final_prob)

                # Update cluster assignment based on the calculated probability
                init_data[i] = np.random.choice(len(norm_prob), 1, p=norm_prob)[0]

                # Computer log-likelihood
                performance.append(np.log(np.sum(final_prob)))

            total_performance.append(np.sum(performance))

        return init_data, mu, sigma, total_performance

    # Display actual parameters
    def display_actual(self):
        print("************ Actual Parameters ************")
        for k in range(self.K):
            print("Cluster Id: {}, Mixture: {:.2f}, Mu: {}, Sigma: {}".format(k, self.actual_mixture[k],
                                                                              self.actual_mu[k],
                                                                              self.actual_sigma[k]))

    # Display estimated parameters
    def display_estimated(self, data, init_cluster, mu, sigma):
        print("************ Estimated Parameters ************")
        cluster_type = Counter(init_cluster)
        for ix, (k, v) in enumerate(sorted(cluster_type.items())):
            print(
                "Cluster Id: {}, Mixture: {:.3f}, Mu: {:.3f}, Sigma: {:.2f}".format(ix,
                                                                                    v / len(data),
                                                                                    mu[ix],
                                                                                    sigma[ix]))

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
    total_iteration = args.iteration
    A = args.alpha
    max_range = args.max_range
    prior = args.prior

    # define model
    model = MixtureModel(K, total_iteration, A, max_range, prior)

    # generate data
    data = model.generate_data()

    # uniform cluster assignment
    init_cluster = model.initialize_cluster(data)

    # gibbs sampling
    init_cluster, mu, sigma, total_performance = model.fit(data, init_cluster)

    # display estimated parameters
    model.display_estimated(data, init_cluster, mu, sigma)

    # display performance plot
    model.display_plot(total_performance)


if __name__ == "__main__":
    main()
