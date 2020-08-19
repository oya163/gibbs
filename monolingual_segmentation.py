# -*- coding: utf-8 -*-
"""
    Description: Unsupervised Monolingual Word Segmentation
    It is based on goldwater-etal-2006-contextual
    https://www.aclweb.org/anthology/P06-1085/
    Base distribution - Geometric distribution

    This program is not complete and in its naive version
    Still under construction
"""

import numpy as np
import matplotlib.pyplot as plt
import grapheme
import pickle

import argparse
from mpmath import gamma

np.random.seed(163)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Mixture Model")
    parser.add_argument("-k", "--cluster", dest="cluster", type=int,
                        default=3, metavar="INT", help="Cluster size [default:3]")
    parser.add_argument("-d", "--data_points", dest="data_points", type=int,
                        default=-1, metavar="INT", help="Total data points [default:-1]")
    parser.add_argument("-i", "--iteration", dest="iteration", type=int,
                        default=5, metavar="INT", help="Iterations [default:50]")
    parser.add_argument("-a", "--alpha", dest="alpha", type=float,
                        default=0.9, metavar="FLOAT", help="Alpha")
    parser.add_argument("-m", "--method", dest="method", type=str, default='collapsed',
                        choices=['mle', 'nig', 'collapsed'], metavar="STR", help="Method Selection [default:collapsed]")
    parser.add_argument('-t', "--testing", default=False, action="store_true", help="For testing purpose only")
    parser.add_argument('-w', "--word", default="नेपालको", metavar="STR", help="Input testing word")
    args = parser.parse_args()
    return args


class MixtureModel:
    def __init__(self, K, A, total_iteration, method):
        # Number of cluster
        self.K = K
        self.A = A
        self.total_iteration = total_iteration
        self.method = method
        self.alpha_0 = 1.0
        self.beta_0 = 2.0

    # Read file
    def read_corpus(self, file_path='national_small.txt'):
        with open(file_path) as f:
            corpus = f.read().split()
            corpus_len = len(corpus)
            print("Corpus length", corpus_len)
        return corpus

    # Single split at each possible boundary
    def split_over_length(self, word):
        split_list = []
        for n in range(1, grapheme.length(word) + 1):
            # split_list.append((word[:n], word[n:len(word)]))
            split_list.append((grapheme.slice(word, 0, n), grapheme.slice(word, n, grapheme.length(word))))
        return split_list

    # Single split at each possible boundary
    def geometric_split(self, word, prob):
        split_point = set(np.random.geometric(prob, size=len(word)))
        split_list = []
        for each in split_point:
            split_list.append((grapheme.slice(word, 0, each), grapheme.slice(word, each, grapheme.length(word))))

        # for n in range(1, grapheme.length(word) + 1):
        # split_list.append((word[:n], word[n:len(word)]))
        # split_list.append((grapheme.slice(word, 0, n), grapheme.slice(word, n, grapheme.length(word))))
        return split_list

    def generate_data(self):
        sent = "नेपाली सिनेमा र एकाध नाटकमा समेत विगत तीस वर्षदेखि क्रियाशील कलाकार राजेश हमाल सिनेमा क्षेत्रका " \
               "महानायक हुन् वा होइनन् भन्नेबारे त्यस क्षेत्रमा रुचि राख्नेहरूबीच तात्तातो बहस चल्यो । "
        sent = sent.split()

        sent = self.read_corpus()

        stem_list = []
        suffix_list = []
        stem_set = set()
        suffix_set = set()
        words = []

        for each in sent:
            # splits = self.geometric_split(each, prob=0.5)
            splits = self.split_over_length(each)
            words.extend(splits)
            for each_split in splits:
                if each_split[0]:
                    stem_list.append(each_split[0])
                if each_split[1]:
                    suffix_list.append(each_split[1])

        print("Length of stem: {} and suffix: {}".format(len(stem_list), len(suffix_list)))
        return words, stem_list, suffix_list

    # Geometric base distribution
    def g0(self, p, k):
        return p * ((1 - p) ** (k - 1))

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
                if orig_data[idx] in new_cluster:
                    new_cluster[orig_data[idx]].append(d)
                else:
                    new_cluster[orig_data[idx]] = [d]
        return new_cluster

    def beta_geometric_posterior(self, x_len, n, sum_of_grapheme):
        alpha_ = self.alpha_0 + n
        beta_ = self.beta_0 + sum_of_grapheme - n
        beta = lambda a, b: (gamma(a) * gamma(b)) / gamma(a + b)
        p = beta(alpha_ + 1, beta_ + x_len - 1) / beta(alpha_, beta_)
        return float(p)

    def fit(self, data, init_data):
        N = len(init_data)
        for itr in range(self.total_iteration):

            for i, x in enumerate(data):
                # Remove data point
                cluster = self.remove_current_data(i, data, init_data)
                x_len = grapheme.length(x)

                # Compress the cluster
                # to prevent from ever increasing cluster id
                keys = sorted(cluster.keys())
                for j in range(0, len(keys)):
                    cluster[j] = cluster.pop(keys[j])

                # Parameters
                cluster_prob = []
                final_prob = []

                # Estimate parameters of each cluster
                for k, v in sorted(cluster.items()):
                    n = len(v)
                    curr_prob = 0.0
                    sum_of_grapheme = sum([grapheme.length(d) for d in v])
                    if self.method == 'mle':
                        # estimate theta using MLE
                        theta_es = n / sum_of_grapheme
                        curr_prob = self.g0(theta_es, x_len)
                    elif self.method == 'collapsed':
                        curr_prob = self.beta_geometric_posterior(x_len, n, sum_of_grapheme)

                    cluster_prob.append(curr_prob)

                    # Count of data points in each cluster
                    likelihood = n / (N + self.A - 1)
                    final_prob.append(likelihood * cluster_prob[-1])

                # Probability of joining new cluster
                final_prob.append((self.A / (N + self.A - 1)) * self.g0(0.5, x_len))

                # Normalize the probability
                norm_prob = final_prob / np.sum(final_prob)

                # Update cluster assignment based on the calculated probability
                init_data[i] = np.random.choice(len(norm_prob), 1, p=norm_prob)[0]

        return init_data

    def clusterize(self, cluster, morpheme_list):
        final_cluster = {}
        for index, id in enumerate(cluster):
            if id in final_cluster:
                final_cluster[id].append(morpheme_list[index])
            else:
                final_cluster[id] = [morpheme_list[index]]

        # for k,v in final_cluster.items():
        #     print(k, v)
        #
        # print("Length of given cluster list = ", len(final_cluster))
        return final_cluster

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

    def get_posterior(self, cluster, morpheme_assignment, initial_list, morpheme):
        index = initial_list.index(morpheme) if morpheme in initial_list else -1
        if index >= 0:
            cluster_id = morpheme_assignment[index]
            n_si = len(cluster[cluster_id])
            return n_si / (len(morpheme_assignment) + self.A)
        else:
            return self.A * self.g0(0.5, grapheme.length(morpheme)) / (len(morpheme_assignment) + self.A)

    def inference(self, st_cluster, sf_cluster, stem_list, suffix_list, given_word):
        # split_list = self.geometric_split(given_word, 0.1)
        split_list = self.split_over_length(given_word)
        stem_cluster = self.clusterize(st_cluster, stem_list)
        suffix_cluster = self.clusterize(sf_cluster, suffix_list)

        final_prob = []
        for stem, suffix in split_list:
            p_stem = self.get_posterior(stem_cluster, st_cluster, stem_list, stem)
            p_suffix = self.get_posterior(suffix_cluster, sf_cluster, suffix_list, suffix)
            final_prob.append(p_stem * p_suffix)

        print("All probable splits")
        for x, y in zip(split_list, final_prob):
            print(x, y)

        # Return splits with max probability
        return split_list[np.argmax(final_prob)], max(final_prob)


def main():
    args = parse_args()

    # Get arguments
    K = args.cluster
    N = args.data_points
    total_iteration = args.iteration
    A = args.alpha
    testing = args.testing
    word = args.word
    method = args.method

    # define model
    model = MixtureModel(K, A, total_iteration, method)

    if not testing:
        # generate data
        customers, stem_list, suffix_list = model.generate_data()

        # uniform cluster assignment
        init_data_stem = model.initialize_cluster(stem_list)
        init_data_suffix = model.initialize_cluster(suffix_list)

        # print(itr)
        st_cluster = model.fit(stem_list, init_data_stem)
        sf_cluster = model.fit(suffix_list, init_data_suffix)

        print("Stem cluster")
        stem_cluster = model.clusterize(st_cluster, stem_list)

        print("Suffix cluster")
        suffix_cluster = model.clusterize(sf_cluster, suffix_list)

        # Save st_cluster, sf_cluster, stem_list, suffix_list
        # Saving the objects:
        with open('segmented.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([st_cluster, sf_cluster, stem_list, suffix_list], f)

    # Restore it
    with open('segmented.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        st_cluster, sf_cluster, stem_list, suffix_list = pickle.load(f)

    # Inference
    best_split, best_prob = model.inference(st_cluster, sf_cluster, stem_list, suffix_list, word)

    print("\nBest split {} {}\n".format(best_split, best_prob))


if __name__ == "__main__":
    main()
