# -*- coding: utf-8 -*-

"""
    Flask based application for monolingual segmentation.
    Given model is based on non-parametric Bayesian model
    with beta-geometric conguate prior

    Date - 09/11/2019
"""

from flask import Flask, render_template, request
from segment import MixtureModel
import segment
import numpy as np
import pickle
import grapheme
from utils import utilities as utilities

application = Flask(__name__)

# args = segment.parse_args()

# Get arguments
K = 3
A = 0.9
N = 5000
total_iteration = 5
alpha_0 = 1.0
beta_0 = 2.0
prob_c = 0.5
method = 'collapsed'
inference = True
evaluation = False
word = 'नेपालमा'
gold_file = './data/gold_standard.txt'
model_filename = './models/segmentation_model.pkl'
input_filename = './data/train.txt'
log_filename = './logs/segmentation.log'
result_filename = './logs/result_file.txt'

logger = utilities.get_logger(log_filename)

# define model
model = MixtureModel(K, A, N, alpha_0, beta_0, prob_c, total_iteration, method, model_filename, input_filename,
                     logger, result_filename)


# Single split at each possible boundary
def split_over_length(word):
    split_list = []
    for n in range(1, grapheme.length(word) + 1):
        # split_list.append((word[:n], word[n:len(word)]))
        split_list.append((grapheme.slice(word, 0, n), grapheme.slice(word, n, grapheme.length(word))))
    return split_list


# Inference
def inference(model, st_cluster, sf_cluster, stem_list, suffix_list, given_word):
    # split_list = geometric_split(given_word, 0.1)
    split_list = split_over_length(given_word)
    stem_cluster = model.clusterize(st_cluster, stem_list)
    suffix_cluster = model.clusterize(sf_cluster, suffix_list)

    final_prob = []
    for stem, suffix in split_list:
        p_stem = model.get_posterior_by_sampling(stem_cluster, st_cluster, stem_list, stem)
        p_suffix = model.get_posterior_by_sampling(suffix_cluster, sf_cluster, suffix_list, suffix)
        final_prob.append(p_stem * p_suffix)

    # Return splits with max probability
    return split_list[np.argmax(final_prob)]


# Restore model
with open('./models/segmentation_model.pkl', 'rb') as f:
    st_cluster, sf_cluster, stem_list, suffix_list = pickle.load(f)


@application.route('/')
def hello():
    return render_template('index.html')


@application.route('/post', methods=['GET', 'POST'])
def post():
    errors = []
    given_word = request.form['input']
    results = inference(model, st_cluster, sf_cluster, stem_list, suffix_list, given_word)

    if request.method == "GET":
        return render_template('index.html')
    else:
        return render_template('index.html', errors=errors, results=results)


if __name__ == "__main__":
    application.run()