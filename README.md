

# Gibbs Sampling ![gibbs_ci](https://github.com/oya163/gibbs/workflows/gibbs_ci/badge.svg) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

These are the outcomes of my learning during my internship (Jun-Aug 2020) in [NAAMII](https://www.naamii.com.np/).
I wrote couple of scripts for finite and infinite mixture models based on Chinese Restaurant Process.
I have shown usages of various conjugate priors and also collapsed sampling methods.

I worked along with Sushil Awale to implement unsupervised monolingual word segmentation based on [goldwater-etal-2006-contextual](https://www.aclweb.org/anthology/P06-1085/)
and [snyder-barzilay-2008-unsupervised](https://www.aclweb.org/anthology/P08-1084/). We came up with monolingual word segmentation with beta-geometric conjugate prior over the length of a given word.

## Basic implementations
- [x] Finite Gaussian Mixture Model
- [x] Infinite Gaussian Mixture Model
- [x] Categorial/Multinomial Mixture Model with Dirichlet prior using MLE
- [x] Categorial/Multinomial Mixture Model with Dirichlet prior using collapsed sampling


## Tasks
- [x] Implement Beta Binomial conjugate prior for word segmentation
- [x] Create Nepali segmentation dataset
- [x] Create simple flask based web app
- [x] Apply CI/CD pipeline
- [x] Deploy on AWS Elastic Beanstalk

## Deployment
Please click [here](http://nepsegment.us-west-2.elasticbeanstalk.com/)