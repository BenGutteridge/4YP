# Fourth Year Project
## Investigating gradient descent in variational inference
Repo for experiment code from project investigating single sample gradient estimation in variational inference. Up to date code is in `March` directory.

#### Abstract
The aim of many machine learning applications is the modelling of data. We may wish to perform
classification of images, clustering of users by preference for recommender systems, or regression
for predicting financial markets; for all of these and limitless other applications, our goal is to learn a
model from some given data. In some cases, we are content with so-called ‘discriminative’ models,
those where we care only about learning the predictor function of interest. Generally though, the
aim is to learn the true underlying model, the system that generated our data and that describes
a real-world process or phenomenon. These are ‘generative’ models, as they not only allow us
to learn systems and make predictions, but also to generate synthetic data similar to that upon
which it was trained (Jebara 2012). We obtain such a generative model by performing Bayesian
posterior inference: choose a distributional family to model the data, and learn the distribution of
model parameters conditioned on that data.
The difficulty in practical inference applications is that in all but the most simple cases, we
must solve an intractable integral. If we cannot evaluate the posterior analytically, we must resort
to approximation, which can be broadly divided into two categories: stochastic methods such as
Monte Carlo, which guarantee convergence but are often slow and inefficient, and deterministic
methods (Bishop 2006). This report focuses on the deterministic approach, also called variational
inference (VI; Jordan et al. 1998; Wainwright and Jordan 2007), which frames posterior inference
as an optimisation problem, and often approximates the posterior much more quickly and efficiently
than by random sampling methods.
As an optimisation problem, we can approach VI with specialised algorithms such as coordinate
ascent variational inference (Bishop 2006), or general-purpose algorithms like stochastic gradient
descent (Robbins and Monro 1951) – in either case, we will find gradients, their calculation and
estimation, at the centre of the method.
In this report we provide a review of VI, with a particular focus on gradient descent algorithms,
estimating accurate gradients, and how this affects the accuracy of posterior approximation and
convergence rates. We also consider Bayesian quadrature (probabilistic numerical integration),
its current applications to VI and possible novel uses, though experimentation in this area was
unfortunately limited due to time constraints and negative results.

## GIFS
Runs used in experiments section of report. Details of individual runs are in report.

### Coordinate ascent VI
![alt text](GMM/figs/cavi.gif)

### Pathwise estimator
![alt text](GMM/figs/pw.gif)

### Gradient descent
![alt text](GMM/figs/gd.gif)

### SGD
![alt text](GMM/figs/sgd.gif)

### Stochastic natural gradient descent
![alt text](GMM/figs/sngd.gif)

<!---
### 9/3/2021
Most up to date, useful code in March directory.

`run_GMM_CAVI_unknown_cov` runs a pure CAVI GMM cluster fitting algo

`run_GMM_GD_m_only_unknown_cov` uses CAVI to pretrain a model (finding distributional parameters for component weights, covs and means), then shifts the Gaussian means randomly and retrains them (distributional parameter of mean only) using TRUE gradient descent  

Next steps:

- Turn GMM into a class, should be a lot cleaner, no more functions with 10+ arguments etc
- Update CAVI for an implementation with known covariance (simpler to do GD/SGD, comparisons)
- Get true GD working for known-cov model for all distributional params


### 28/1/2021
Co-ordinate variational inference for a Gaussian mixture model:
https://colab.research.google.com/drive/1JrLIbMR4OrrBDVjIrJX8tKpMRhM-_8Wd?usp=sharing

`GMM_var_inf_Bishop` contains an updated script for batch GD -- this works better than the script in `GMM`, which I am leaving alone for the time being.

N.b. The above notebook requires `MM_var_inf_Bishop/utils.py`




![alt text](GMM/figs/GMM_components.gif)

VAE Colab notebook from earlier is here:
https://colab.research.google.com/drive/163uMDyCo96dIqItTlN9gW2oGa2zrDOyp?usp=sharing
-->
