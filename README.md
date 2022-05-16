# Reliable Predictive Inference in Time-Series Settings

An important factor to guarantee a responsible use of data-driven systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing prediction sets, which provide an intuitive measure of the limits of predictive performance.

This package contains a Python implementation of Rolling Conformal Inference (Rolling CI) [1] methodology for constructing distribution-free prediction sets. 

# Conformalized Online Learning: Online Calibration Without a Calibration Set [1]

Rolling CI is a method that reliably reports the uncertainty of a target variable response in the time-series setting and provably attains the user-specified coverage level over long-time intervals.

[1] Shai Feldman, Stephen Bates, Yaniv Romano, [“Conformalized Online Learning: Online Calibration Without a Calibration Set”]() 2022.

## Getting Started

This package is self-contained and implemented in python.

Part of the code is a taken from the [oqr](https://github.com/Shai128/oqr) and [mqr](https://github.com/Shai128/mqr) packages. 


### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/shai128/rci.git
```

## Usage


### Rolling CI

Comparisons to competitive methods and can be found in [display_results.ipynb](display_results.ipynb).

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

### Publicly Available Datasets


* [Power](https://archive.ics.uci.edu/ml/datasets/Power+consumption+of+Tetouan+city): Power consumption of Tetouan city Data Set.

* [Energy](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction): Appliances energy prediction Data Set.

* [Traffic](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume): Metro Interstate Traffic Volume Data Set.

* [Wind](https://www.kaggle.com/datasets/l3llff/wind-power): Wind Power in Germany.

* [Power](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries/blob/main/data_prices/Prices_2016_2019_extract.csv): French electricity prices [2].

[2] Margaux Zaffran, Aymeric Dieuleveut, Olivier Féron, Yannig Goude, Julie Josse, [“Adaptive Conformal Predictions for Time Series.”](https://arxiv.org/abs/2202.07282) 2022.
