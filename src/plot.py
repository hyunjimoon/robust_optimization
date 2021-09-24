import pandas as pd
import numpy as np
#from contextPolicy import argmin_lopt_genargmin_lopt_emp, argmin_lpred, argmin_lopt_bar_argmin_lpred
def train_test_err(X, X_new, y_w, y_w_new, y_m, y_m_new, Thetastar):
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
  test_size = 100000
  what1, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X, Theta = None, y = y_w)
  train_df['err_w1'] = lopt_NV(what1, y_w) 
  what2, Thetahat2 = argmin_lopt_emp("lin", "Norm", X, Theta = None, y = y_w)
  train_df['err_w2'] = lopt_NV(what2, y_w) 


  test_df['err_new_w1'] = lopt_NV(argmin_lopt_bar_argmin_lpred("lin", "Norm", X, Theta = Thetahat1, y = None), y_w_new) 
  test_df['err_new_w2'] = lopt_NV(argmin_lopt_emp("lin", "Norm", X, Theta = Thetahat2, y = None), y_w_new) 

  what1, Thetahat1 = argmin_lopt_bar_argmin_lpred("lin", "Norm", X, Theta = None, y = y_m)
  train_df['err_m1'] = lopt_NV(what1, y_w) 
  what2, Thetahat2 = argmin_lopt_emp("lin", "Norm", X,  Theta = None, y = y_m)
  train_df['err_m2'] = lopt_NV(what2, y_w) 

  test_df['err_new_m1'] = lopt_NV(argmin_lopt_bar_argmin_lpred("lin", "Norm", X, Theta = Thetahat1, y = None), y_m_new) 
  test_df['err_new_m2'] = lopt_NV(argmin_lopt_emp("lin", "Norm", X, Theta = Thetahat2, y = None), y_m_new) 
  return train_df, test_df
