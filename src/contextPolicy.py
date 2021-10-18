import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from cmdstanpy import cmdstan_path, CmdStanModel
from sklearn.linear_model import LinearRegression
import jax
from jax.experimental import optimizers
import jax.scipy as jsc
import jax.numpy as jnp
from loss import lopt_NV
from generator import sim_y_barThetax, sim_ThetaXy, P_ybarx, check_random_state, make_low_rank_matrix

# approach 1 if y = None: W(Theta, X) else Thetahat, W(Thetahat, x)
def argmin_lopt_bar_argmin_lpred(link, family, X, Thetahat, y, input_type, alg_type):
    '''
    Compute approach1 optimal solution; separate predict optimize
    Parameters:
         char MClass: model class type specificed with `P_ybarx`
            "lin": y_true = theta' * X
            "quad": y_true = theta' * X^2
          real Thetahat: coefficient vector of size [p, 1] 
            "fixed"
            real value of Theta (computing trainset error)
          np.array X: predictor data of size [n, p] 
          chr alg_type: Solver algorithm
    Returns:
        np.array what: predict then optimize optimal solution of size [n,1]
    '''
    # if not any([y, Theta]):
    #   print("Both Theta and y cannot be None")
    what = np.repeat(np.nan, len(X))
    if input_type == "Theta":
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat)
        return what
    else:
        Thetahat = argmin_lpred(link, family, X, y, alg_type = alg_type)
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat)
        return what, Thetahat

# approach 2
def argmin_lopt_emp(link, family, X, Thetahat, y, input_type, Thetastar, search_type = "grid"):
    '''
    Compute approach2 optimal solution; empirical objective minimization
    Parameters:
        char link: model class type specificed with
            "lin": y_true = Theta' * X
            "quad": y_true = x.transpose() @ Theta_mat @ x + Theta @ x
        char family: distribution type
        array X: predictor data of size [n, p]
        vector Theta: estimated 
        array y: outcome data of size [n, 1]

        char search_type: `grid`, `autodiff`
    Returns:
        array what: optimal solution for each x of size [n,1]
    '''
    # if not any([y, Theta]):
    #   print("Both Theta and y cannot be None")
    what = np.repeat(np.nan, len(X))
    if input_type == "Theta":
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat)
        return what
    else:
        if search_type == "grid":
            def lopt_sim(link, family, X, y, Thetahat):
                '''
                Loss when function is known upto P_Y|X^theta, but not w
                '''
                lopt = 0
                for i in range(len(X)):
                    what_x = argmin_lopt_gen(link, family, X[i], Thetahat)
                    lopt -= (5 * min(what_x, y[i]) - 1 * what_x)
                return lopt/len(X)
            theta1 = np.linspace(Thetastar[0]*.8, Thetastar[0]*1.2, num = 30)
            theta2 = np.linspace(Thetastar[1]*.8, Thetastar[1]*1.2, num = 30)
            theta1, theta2 = np.meshgrid(theta1, theta2)
            M = 1000000
            for Theta in zip(theta1.ravel(), theta2.ravel()):
                #lopt_NV(what, y[i], profit, cost) 'numpy.float64' object cannot be interpreted as an integer
                if lopt_sim(link, family, X, y, Theta) < M:
                    Thetahat = Theta
                    M = lopt_sim(link, family, X, y, Thetahat)
            print("loss", M, "Thetahat2", Thetahat)
            print("min_loss", lopt_sim(link, family, X, y, Thetastar), "Thetastar", Thetastar)
        elif search_type == "autodiff":
            def lopt_fn(param, X, y): #param = p*1, X:n*p, x= 1*p
                '''
                Loss when function is known upto w(x, P_Y|X^theta)
                '''
                lopt = 0
                for i in range(len(X)):
                    what_x = jsc.stats.norm.ppf(loc = param @ X[i], q =0.8) #TODO
                    lopt += - (5 * min(what_x, y[i]) - 1 * what_x)
                return lopt
            Theta = Thetastar
            lr = 10
            opt_init, opt_update, get_params = optimizers.rmsprop(lr)
            opt_state = opt_init(Theta)
            for i in range(30): # TODO cvg diag.
                loss, grads = jax.value_and_grad(lopt_fn)(get_params(opt_state), X, y)
                opt_state = opt_update(i, grads, opt_state)
                print("loss", loss, "Thetastar", Thetastar, "Thetahat", get_params(opt_state), "grads", grads)
            Thetahat = get_params(opt_state)
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat)
        return what, Thetahat

# plug-in opt.solver
def argmin_lopt_gen(link, family, x, Theta, closed_type = "W_Theta", **kwargs):
    '''
    Compute optimal solution for each Theta and X assuming $Y|X \sim P_{\Theta}$
    Parameters:
        char link: "log", "linear", "quadratic" (, "logit") relation between eta and mu
        char family: "Normal", "Exp", "Cauchy"
        array Theta: model parameter of size [p, 1]
        array x: predictor data of size [1, p]
        function lopt: loss function, given w(x), y, X, outputs loss
    Returns:
        array w: optimal solution for each x of size [n,1]
    '''
    sigma_y = 1
    if closed_type == "P_ybarx":      
        y_sim = P_ybarx(link, family, x, Theta, sigma_y)
        data = {'y': list(y_sim), 'n': len(y_sim)}
        data = {**data, **kwargs}
        sm = cmdstanpy.CmdStanModel(stan_file="/content/drive/MyDrive/Colab Notebooks/robust_optimization/src/stan/optW_lopt_NV.stan")
        W_Thetax = sm.optimize(data).stan_variable('w')
    elif closed_type == "W_Theta":      
        W_Thetax = np.dot(x, Theta) + norm.ppf(.8) * sigma_y
    return W_Thetax
def argmin_lpred(link, family, X, y, alg_type = "OLS"):
    '''
    Solve parameter that best predicts y given X assuming $y\sim N(X * Theta, 1)$
    Parameters:
        char link: model class type specificed with `P_ybarx` which is user-defined
             input for `argmin_lopt_gen`, `argmin_lopt_emp`
             SHOULD comply with "P_ybarx_f{link}.stan"
             "lin": y_true = Theta' * X
             "quad": y_true = Theta' * X^2
        char family: distribution type specificed e.g. "Normal", "Cauchy"
        np.array X: predictor data of size [n, p]
        np.array y: outcome data of size [n, 1]
        chr alg_type: Solver algorithm
             "OLS": (xx')^x'beta                                                                                                                                                                     
             "sample": MAP with improper uniform priors (undeclared) in stan file with HMC
             "optimize": Maximum likelihood estimate () with "lbfgs", "bfgs", "newton"
             "varaiational": variational inference with ADVI, RVI
             https://mc-stan.org/cmdstanpy/examples/Maximum%20Likelihood%20Estimation.html
             https://mc-stan.org/cmdstanpy/examples/Variational%20Inference.html
    Returns:
        np.array Thetahat: optimal parameter value of size [p, 1]
    '''

    if alg_type == "OLS":
        reg = LinearRegression(fit_intercept = False).fit(X, y)
        Thetahat = reg.coef_
    else:
        lpred_stan = os.path.join('stan', 'optTheta_lpred.stan')
        data = {'X':X.tolist(), 'y':y, 'n': len(X), 'p': len(X[1]), 'sigma_y': 1}
        sm = CmdStanModel(stan_file=lpred_stan) # true Theta: 2, 2
        if alg_type == "sample":
            Thetahat = np.mean(sm.sample(data).stan_variable('beta'), axis =0) # 2.00, 1.98
        elif alg_type == "optimize":
            Thetahat = sm.optimize(data).stan_variable('beta') # 2.00 , 1.98
        elif alg_type == "variational":
            Thetahat = np.mean(sm.sample(data).stan_variable('beta'), axis =0) # 1.99, 1.99
    #reg = LinearRegression(fit_intercept = False).fit(X, y)
    return Thetahat #reg.coef_
