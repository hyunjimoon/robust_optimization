# Robust Optimization

Random variable view of optimal solution and its value

# Theory for consistency and convergence rate
Theoretical relation would be M estimator.

# Code
`generator.py`: data are generated and you might want to becareful about their scale and signs especially if you are using certain closed form solutions that is based on implicit assumptions (e.g. nonnegative demand).

`contextPolicy.py`: two approaches (`argmin_lopt_bar_argmin_lpred`, `argmin_lopt_emp`) with different loss functions which returns optimal parameter and shared optimization solver that returns optimal solution given parameter and predictor.

`plot.py`: retrun loss values for fixed X_train, X_test, y_train, y_test, Thetastar

# Experiment
Observe distribution of measure value such as true overall expected profit evaluated at estimated solution E_{true P}[Cost(y, W_thetahat(x))], by repeating the experiment thousand times and computing their mean and variance. Experiment setup is similar in https://arxiv.org/abs/1810.02905 though the traget random variable is coverage of bounds in that example.
