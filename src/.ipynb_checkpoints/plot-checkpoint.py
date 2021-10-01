import pandas as pd
import numpy as np
from contextPolicy import argmin_lopt_bar_argmin_lpred, argmin_lopt_emp
from loss import lopt_NV

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
    what1_train, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_train, Theta = None, y = y_train, input_type = "y", alg_type = "optimize")
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

def train_test_err_dep(X_train, X_test, y_w_train, y_w_test, y_m_train, y_m_test, Thetastar):
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
    what1, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_train, Theta = None, y = y_w_train, input_type = "y", alg_type = "optimize")
    what2, Thetahat2 = argmin_lopt_emp("lin", "Norm", X_train, Theta = None, y = y_w_train, input_type = "y", Thetastar = Thetastar)
    #search_type = search_type? TypeError: cannot unpack non-iterable NoneType object

    # Well-specified test
    what1_test = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_test, Theta = Thetahat1, y = None, input_type = "Theta")
    what2_test = argmin_lopt_emp("lin", "Norm", X_test, Theta = Thetahat2, y = None, input_type = "Theta", Thetastar = Thetastar)
    
    # save
    train_df['w1'] = what1
    train_df['w2'] = what2
    test_df['w1'] = what1_test
    test_df['w2'] = what2_test
    
    # todo pickle, hash
    # train_df.to_pickle(f"train_{len(train_df)}.csv")
    # test_df.to_pickle(f"train_{len(test_df)}.csv")
    # pd.read_pickle(f"train_{len(train_df)}.csv")
    # pd.read_pickle(f"test_{len(test_df)}.csv")
    
#     # Mis-specified train
    what1, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_train, Theta = None, y = y_m_train, input_type = "y")
    what2, Thetahat2 = argmin_lopt_emp("lin", "Norm", X_train,  Theta = None, y = y_m_train, input_type = "y", Thetastar = Thetastar)

    # Mis-specified test
    what1_test = argmin_lopt_bar_argmin_lpred("lin", "Norm", X_test, Theta = Thetahat1, y = None, input_type = "Theta")

    what2_test = argmin_lopt_emp("lin", "Norm", X_test, Theta = Thetahat2, y = None, input_type = "Theta", Thetastar = Thetastar)

    # save    
    train_df_m['wm1'] = what1
    train_df_m['wm2'] = what2
    train_df_m['wm1'] = what1_test
    train_df_m['wm2'] = what2_test
    return train_df, test_df, train_df_m, test_df_m