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

def train_test_w(X_train, X_test, y_train, y_test, Thetastar):
    '''
    Receive feature and outcome for train and testset with ground_truth
    Parameters:
      char specification type: 
            "well": generated with y_true = Theta' * X
            "miss": generated with exp(theta*x)
      real sigma_y: noise scale
    Returns:
      train, test error
    '''
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()   
    train_df['meantrue'] = np.dot(X_train, Thetastar)
    test_df['meantrue'] = np.dot(X_test, Thetastar)
    train_df['X0'] = X_train[:,0]
    train_df['X1'] = X_train[:,1]
    test_df['X0'] = X_test[:,0]
    test_df['X1'] = X_test[:,1]
    
    # Well-specified train
    what1_train, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_train, Theta = None, y = y_train, input_type = "y")
    what2_train, Thetahat2 = argmin_lopt_emp("lin", "Norm", X_train, Theta = None, y = y_train, input_type = "y", Thetastar = Thetastar)
    #search_type = search_type? TypeError: cannot unpack non-iterable NoneType object

    # Well-specified test
    what1_test = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_test, Theta = Thetahat1, y = None, input_type = "Theta")
    what2_test = argmin_lopt_emp("lin", "Norm", X_test, Theta = Thetahat2, y = None, input_type = "Theta", Thetastar = Thetastar)
    
    # save
    train_df['w1'] = what1_train
    train_df['w2'] = what2_train
    test_df['w1'] = what1_test
    test_df['w2'] = what2_test
    
    # todo pickle, hash
    # train_df.to_pickle(f"train_{len(train_df)}.csv")
    # test_df.to_pickle(f"train_{len(test_df)}.csv")
    # pd.read_pickle(f"train_{len(train_df)}.csv")
    # pd.read_pickle(f"test_{len(test_df)}.csv")

    return train_df, test_df

# approach 1 if y = None: W(Theta, X) else Thetahat, W(Thetahat, x)
def argmin_lopt_bar_argmin_lpred(link, family, X, Theta, y, input_type, alg_type = "OLS"):
    '''
    Compute approach1 optimal solution; separate predict optimize
    Parameters:
         char MClass: model class type specificed with `P_ybarx`
            "lin": y_true = theta' * X
            "quad": y_true = theta' * X^2
          real Theta: coefficient vector of size [p, 1] 
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
            what[i] = argmin_lopt_gen(link, family, X[i], Theta)
        return what
    else:
        Thetahat1 = argmin_lpred(link, family, X, y, alg_type = alg_type)
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat1)
        return what, Thetahat1

# approach 2
def argmin_lopt_emp(link, family, X, Theta, y, input_type, Thetastar = None, search_type = "grid"):
    '''
    Compute approach2 optimal solution; empirical objective minimization
    Parameters:
        char link: model class type specificed with
            "lin": y_true = Theta' * X
            "quad": y_true = x.transpose() @ Theta_mat @ x + Theta @ x
        char family: distribution type
        array y: outcome data of size [n, 1]
        array X: predictor data of size [n, p]
        char search_type: `grid`, `autodiff`
    Returns:
        array what: optimal solution for each x of size [n,1]
    '''
    # if not any([y, Theta]):
    #   print("Both Theta and y cannot be None")
    what = np.repeat(np.nan, len(X))
    if input_type == "Theta":
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Theta)
        return what
    else:
        if search_type == "grid":
            def lopt_sim(link, family, X, Theta):
                '''
                Loss when function is known upto P_Y|X^theta, but not w
                '''
                lopt = 0
                for i in range(len(X)):
                    what_x = argmin_lopt_gen(link, family, X[i], Theta)
                    lopt -= (5 * min(what_x, y[i]) - 1 * what_x)
                return lopt
            theta1 = np.linspace(Thetastar[0]*.85, Thetastar[0]*1.15, num = 50)
            theta2 = np.linspace(Thetastar[1]*.85, Thetastar[1]*1.15, num = 50)
            theta1, theta2 = np.meshgrid(theta1, theta2)
            M = 1000000
            for Theta in zip(theta1.ravel(), theta2.ravel()):
                #lopt_NV(what, y[i], profit, cost) 'numpy.float64' object cannot be interpreted as an integer
                if lopt_sim(link, family, X, Theta) < M:
                    Thetahat2 = Theta
                    M = lopt_sim(link, family, X, Theta)
                    print("loss", lopt_sim(link, family, X, Theta),  "Thetastar", Thetastar, "theta", Thetahat2)
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
            Thetahat2 = get_params(opt_state)
        for i in range(len(X)):
            what[i] = argmin_lopt_gen(link, family, X[i], Thetahat2)
        return what, Thetahat2

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
    return Thetahat
