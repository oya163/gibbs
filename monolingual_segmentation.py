# -*- coding: utf-8 -*-

# Description: Unsupervised Monolingual Word Segmentation
# It is based on goldwater-etal-2006-contextual
# https://www.aclweb.org/anthology/P06-1085/
# Base distribution - Geometric distribution


import regex as re
import numpy as np
import matplotlib.pyplot as plt
import grapheme
import pickle
import math
import argparse
import utilities as utilities
from mpmath import gamma
import sys
import csv

np.random.seed(163)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Mixture Model")
    parser.add_argument("-k", "--cluster", dest="cluster", type=int,
                        default=3, metavar="INT", help="Cluster size [default:3]")
    parser.add_argument("-d", "--data_points", dest="data_points", type=int,
                        default=5000, metavar="INT", help="Total data points [default:5000]")
    parser.add_argument("-i", "--iteration", dest="iteration", type=int,
                        default=5, metavar="INT", help="Iterations [default:50]")
    parser.add_argument("-a", "--alpha", dest="alpha", type=float,
                        default=0.9, metavar="FLOAT", help="Alpha")
    parser.add_argument("--alpha_0", dest="alpha_0", type=float,
                        default=1.0, metavar="FLOAT", help="Beta Geometric Alpha")
    parser.add_argument("--beta_0", dest="beta_0", type=float,
                        default=2.0, metavar="FLOAT", help="Beta Geometric Beta")
    parser.add_argument("-p", "--prob_c", dest="prob_c", type=float,
                        default=0.5, metavar="FLOAT", help="Probability of joining new cluster")
    parser.add_argument("-m", "--method", dest="method", type=str, default='collapsed',
                        choices=['mle', 'nig', 'collapsed'], metavar="STR", help="Method Selection [default:collapsed]")
    parser.add_argument("--input_filename", dest="input_filename", type=str, default='train.txt',
                        metavar="PATH", help="Input Filename [default:national_very_small.txt]")
    parser.add_argument("-f", "--model_filename", dest="model_filename", type=str, default='segmented',
                        metavar="STR", help="File name [default:segmented]")
    parser.add_argument("-l", "--log_filename", dest="log_filename", type=str, default='segmentation.log',
                        metavar="PATH", help="File name [default:segmentation.log]")
    parser.add_argument('-t', "--inference", default=False, action="store_true", help="For inference purpose only")
    parser.add_argument('-w', "--word", default="नेपालको", metavar="STR", help="Input inference word")
    parser.add_argument('-e', "--evaluation", default=False, action="store_true", help="For evaluation purpose only")
    parser.add_argument("-g", "--gold_file", dest="gold_file", type=str,
                        default='gold_standard.txt', required='--evaluate' in sys.argv,
                        metavar="PATH", help="File name [default:gold_standard.txt]")
    args = parser.parse_args()
    return args


def process(text):
    text = re.sub(r'\([^)]*\)', r'', text)
    text = re.sub(r'\[[^\]]*\]', r'', text)
    text = re.sub(r'<[^>]*>', r'', text)
    text = re.sub(r'[!।,\']', r'', text)
    text = re.sub(r'[०१२३४५६७८९]', r'', text)
    text = text.replace(u'\ufeff', '')
    text = text.replace(u'\xa0', u' ')
    text = re.sub(r'( )+', r' ', text)
    return text


# Single split at each possible boundary
def split_over_length(word):
    split_list = []
    for n in range(1, grapheme.length(word) + 1):
        # split_list.append((word[:n], word[n:len(word)]))
        split_list.append((grapheme.slice(word, 0, n), grapheme.slice(word, n, grapheme.length(word))))
    return split_list


# Single split at each possible boundary
def geometric_split(word, prob):
    split_point = set(np.random.geometric(prob, size=len(word)))
    split_list = []
    for each in split_point:
        split_list.append((grapheme.slice(word, 0, each), grapheme.slice(word, each, grapheme.length(word))))

    # for n in range(1, grapheme.length(word) + 1):
    # split_list.append((word[:n], word[n:len(word)]))
    # split_list.append((grapheme.slice(word, 0, n), grapheme.slice(word, n, grapheme.length(word))))
    return split_list


# Geometric base distribution
def g0(p, k):
    return p * ((1 - p) ** (k - 1))


# Remove given data from cluster
def remove_current_data(index, input_data, orig_data):
    new_cluster = {}
    for idx, d in enumerate(input_data):
        # Skip the given index
        if idx != index:
            if orig_data[idx] in new_cluster:
                new_cluster[orig_data[idx]].append(d)
            else:
                new_cluster[orig_data[idx]] = [d]
    return new_cluster


class MixtureModel:
    def __init__(self, K, A, N, alpha_0, beta_0, prob_c, total_iteration, method, model_filename, input_filename, logger):
        # Number of cluster
        self.K = K
        self.A = A
        self.N = N
        self.total_iteration = total_iteration
        self.method = method
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.prob_c = prob_c
        self.model_filename = model_filename
        self.input_filename = input_filename
        self.logger = logger

        self.logger.info("\n======================HYPERPARAMETERS=============================\n")
        self.logger.info("Initial cluster size : {}".format(self.K))
        self.logger.info("CRP alpha : {}".format(self.A))
        self.logger.info("Number of words : {}".format(self.N))
        self.logger.info("Number of iteration : {}".format(self.total_iteration))
        self.logger.info("Selected method : {}".format(self.method))
        self.logger.info("Beta Geometric alpha : {}".format(self.alpha_0))
        self.logger.info("Beta Geometric beta : {}".format(self.beta_0))
        self.logger.info("Probability of joining new cluster : {}".format(self.prob_c))
        self.logger.info("Model filename : {}".format(self.model_filename))
        self.logger.info("Training filename : {}".format(self.input_filename))
        self.logger.info("\n==================================================================\n")

    # Read file
    def read_corpus(self):
        with open(self.input_filename) as f:
            text = process(f.read())
            corpus = text.split()[:self.N]
            corpus_len = len(corpus)
            self.logger.info("Corpus length : {}".format(corpus_len))
        return corpus

    # Generates data, splitting into stem/suffix
    def generate_data(self):
        # sent = "नेपाली सिनेमा र एकाध नाटकमा समेत विगत तीस वर्षदेखि क्रियाशील कलाकार राजेश हमाल सिनेमा क्षेत्रका " \
        #        "महानायक हुन् वा होइनन् भन्नेबारे त्यस क्षेत्रमा रुचि राख्नेहरूबीच तात्तातो बहस चल्यो । "
        # sent = sent.split()

        sent = self.read_corpus()

        stem_list = []
        suffix_list = []
        stem_set = set()
        suffix_set = set()
        words = []

        for each in sent:
            splits = geometric_split(each, prob=0.5)
            # splits = split_over_length(each)
            words.extend(splits)
            for each_split in splits:
                if each_split[0]:
                    stem_list.append(each_split[0])
                if each_split[1]:
                    suffix_list.append(each_split[1])

        self.logger.info("Length of stem: {} and suffix: {}".format(len(stem_list), len(suffix_list)))
        return words, stem_list, suffix_list

    # Random assignment by generating random number
    # between 0 and given number of cluster
    def initialize_cluster(self, data):
        init_data = []
        for each in data:
            init_data.append(np.random.randint(0, self.K))
        return init_data

    # Beta geometric conjuate prior
    def beta_geometric_posterior(self, x_len, n, sum_of_grapheme):
        alpha_ = self.alpha_0 + n
        beta_ = self.beta_0 + sum_of_grapheme - n
        beta = lambda a, b: (gamma(a) * gamma(b)) / gamma(a + b)
        p = beta(alpha_ + 1, beta_ + x_len - 1) / beta(alpha_, beta_)
        return float(p)

    # Helper fit function
    def _fit(self, data, init_data):
        H = len(init_data)
        performance = []
        for i, x in enumerate(data):
            # Remove data point
            cluster = remove_current_data(i, data, init_data)
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
                    curr_prob = g0(theta_es, x_len)
                elif self.method == 'collapsed':
                    curr_prob = self.beta_geometric_posterior(x_len, n, sum_of_grapheme)

                cluster_prob.append(curr_prob)

                # Count of data points in each cluster
                likelihood = n / (H + self.A - 1)
                final_prob.append(likelihood * cluster_prob[-1])

            # Probability of joining new cluster
            final_prob.append((self.A / (H + self.A - 1)) * g0(self.prob_c, x_len))

            # Normalize the probability
            norm_prob = final_prob / np.sum(final_prob)

            # Update cluster assignment based on the calculated probability
            init_data[i] = np.random.choice(len(norm_prob), 1, p=norm_prob)[0]

            # Computer log-likelihood
            performance.append(np.log(np.sum(final_prob)))

        return init_data, performance

    # Main fit function
    def fit(self, data, init_data, data_list):
        stem_list, suffix_list = data_list
        total_performance = [[] for x in range(2)]
        best_performance = -math.inf
        for itr in range(self.total_iteration):
            self.logger.info("Training iteration: {}".format(itr))
            st_cluster, st_performance = self._fit(data[0], init_data[0])
            sf_cluster, sf_performance = self._fit(data[1], init_data[1])

            self.logger.info("Cluster size : {}\n".format(set(st_cluster)))

            curr_performance = (np.sum(st_performance) + np.sum(sf_performance)) / 2
            total_performance[0].append(np.sum(st_performance))
            total_performance[1].append(np.sum(sf_performance))

            if curr_performance > best_performance:
                best_performance = curr_performance
                save_filename = self.model_filename + '.pkl'
                self.logger.info("Best model saved to {}".format(save_filename))
                with open(save_filename, 'wb') as f:
                    pickle.dump([st_cluster, sf_cluster, stem_list, suffix_list], f)

        return total_performance

    # Accumulate the data into cluster {key: value} pairs
    def clusterize(self, cluster, morpheme_list):
        final_cluster = {}
        for index, id in enumerate(cluster):
            if id in final_cluster:
                final_cluster[id].append(morpheme_list[index])
            else:
                final_cluster[id] = [morpheme_list[index]]

        # for k,v in final_cluster.items():
        #     self.logger.info(k, v)
        #
        # self.logger.info("Length of given cluster list = ", len(final_cluster))
        return final_cluster

    # Display log likelihood plot
    def display_plot(self, total_performance):
        plt.figure(figsize=(10, 5))
        plt.plot(total_performance)
        plt.title("Log-Likelihood")
        plt.xlabel("Number of Iteration")
        plt.ylabel("log p(x_i,z_i|X_-i, Z_-i, theta)")
        plt.xticks([i for i in range(self.total_iteration)])
        plt.show()

    # Inference
    # Get posterior probability based on the cluster assignment
    # of given morpheme, assumption is a morpheme is assigned to only one cluster
    def get_posterior_by_index(self, cluster, morpheme_assignment, initial_list, morpheme):
        index = initial_list.index(morpheme) if morpheme in initial_list else -1
        if index >= 0:
            cluster_id = morpheme_assignment[index]
            n_si = len(cluster[cluster_id])
            return n_si / (len(morpheme_assignment) + self.A)
        else:
            return self.A * g0(self.prob_c, grapheme.length(morpheme)) / (len(morpheme_assignment) + self.A)

    # Inference
    # Get posterior probability based on sampling among the cluster assignment
    # of given morpheme, because a morpheme can be assigned to multiple clusters
    def get_posterior_by_sampling(self, cluster, morpheme_assignment, initial_list, morpheme):
        initial_list = np.array(initial_list)
        indices = np.where(initial_list == morpheme)[0].tolist()
        L = len(morpheme_assignment)
        prob = []
        if indices:
            for index in indices:
                cluster_id = morpheme_assignment[index]
                n_si = len(cluster[cluster_id])
                prob.append(n_si / (L + self.A))

            # Sampling from the assigned clusters
            norm_prob = prob / np.sum(prob)
            prob_index = np.random.choice(len(norm_prob), 1, p=norm_prob)[0]
            return prob[prob_index]
        else:
            return self.A * g0(0.5, grapheme.length(morpheme)) / (L + self.A)

    # Inference
    def inference(self, st_cluster, sf_cluster, stem_list, suffix_list, given_word):
        # split_list = geometric_split(given_word, 0.1)
        split_list = split_over_length(given_word)
        stem_cluster = self.clusterize(st_cluster, stem_list)
        suffix_cluster = self.clusterize(sf_cluster, suffix_list)

        final_prob = []
        for stem, suffix in split_list:
            p_stem = self.get_posterior_by_sampling(stem_cluster, st_cluster, stem_list, stem)
            p_suffix = self.get_posterior_by_sampling(suffix_cluster, sf_cluster, suffix_list, suffix)
            final_prob.append(p_stem * p_suffix)

        self.logger.info("\n======================INFERENCE=============================\n")
        self.logger.info("All probable splits")
        for x, y in zip(split_list, final_prob):
            self.logger.info("{} {}".format(x, y))

        # Return splits with max probability
        return split_list[np.argmax(final_prob)], max(final_prob)

    # Evaluate
    def evaluate(self, st_cluster, sf_cluster, stem_list, suffix_list, gold_file):
        # Read gold file and collect only words
        hit = 0
        insert = 0
        delete = 0
        with open(gold_file, 'r') as f, open('result_file.txt', 'w') as g:
            reader = csv.reader(f, delimiter='\t')
            for word, morphemes in reader:
                # Do this process for each word
                split_list = split_over_length(word)
                stem_cluster = self.clusterize(st_cluster, stem_list)
                suffix_cluster = self.clusterize(sf_cluster, suffix_list)

                final_prob = []
                for stem, suffix in split_list:
                    p_stem = self.get_posterior_by_sampling(stem_cluster, st_cluster, stem_list, stem)
                    p_suffix = self.get_posterior_by_sampling(suffix_cluster, sf_cluster, suffix_list, suffix)
                    final_prob.append(p_stem * p_suffix)

                best_split = split_list[np.argmax(final_prob)]
                pred_stem, pred_suffix = best_split[0], best_split[1]
                gold_stem, gold_suffix = morphemes.split()[0], morphemes.split()[1]
                pred_stem_len = grapheme.length(pred_stem)
                gold_stem_len = grapheme.length(gold_stem)
                if pred_stem_len == gold_stem_len:
                    hit += 1
                elif pred_stem_len < gold_stem_len:
                    insert += 1
                elif pred_stem_len > gold_stem_len:
                    delete += 1

                note = word + '\t' + morphemes + '\t' + best_split[0] + ' ' + best_split[1] + '\n'

                g.write(note)

            # Return prec, recall, f1
            prec = hit/(hit + insert)
            recall = hit/(hit + delete)
            fscore = (2*hit) / ((2*hit)+insert+delete)
            return prec, recall, fscore


def main():
    args = parse_args()

    # Get arguments
    K = args.cluster
    N = args.data_points
    total_iteration = args.iteration
    A = args.alpha
    alpha_0 = args.alpha_0
    beta_0 = args.beta_0
    prob_c = args.prob_c
    inference = args.inference
    evaluation = args.evaluation
    gold_file = args.gold_file
    word = args.word
    method = args.method
    model_filename = args.model_filename
    input_filename = args.input_filename
    log_filename = args.log_filename

    logger = utilities.get_logger(log_filename)

    # define model
    model = MixtureModel(K, A, N, alpha_0, beta_0, prob_c, total_iteration, method, model_filename, input_filename, logger)

    if not inference and not evaluation:
        logger.info("\n======================TRAINING=============================\n")
        # generate data
        customers, stem_list, suffix_list = model.generate_data()

        # uniform cluster assignment
        init_data_stem = model.initialize_cluster(stem_list)
        init_data_suffix = model.initialize_cluster(suffix_list)

        total_performance = model.fit((stem_list, suffix_list),
                                      (init_data_stem, init_data_suffix),
                                      (stem_list, suffix_list))

        # Stem/Suffix display plot
        model.display_plot(total_performance[0])
        model.display_plot(total_performance[1])

        # If training, then perform evaluation and inference
        evaluation = True
        inference = True

    if evaluation:
        # Restore model
        with open(model_filename + '.pkl', 'rb') as f:
            st_cluster, sf_cluster, stem_list, suffix_list = pickle.load(f)

        # Evaluation
        prec, rec, fscore = model.evaluate(st_cluster, sf_cluster, stem_list, suffix_list, gold_file)

        logger.info("Precision: {:.3f}, Recall: {:.3f}, F-score: {:.3f}".format(prec, rec, fscore))

    if inference:
        # Inference
        best_split, best_prob = model.inference(st_cluster, sf_cluster, stem_list, suffix_list, word)

        logger.info("Best split {} {}\n".format(best_split, best_prob))


if __name__ == "__main__":
    main()
