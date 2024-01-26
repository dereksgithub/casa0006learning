import pandas as pd

class diy_bagging():
    """
    

    Parameters
    """
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0

    def diy_bagging(input_df, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        """
        
        
        Parameters
        ----------
        input_df: pandas.DataFrame
        target_df, 
        n_estimators=10, 
        max_samples=1.0, 
        max_features=1.0, 
        bootstrap=True, 
        bootstrap_features=False, 
        oob_score=False, 
        warm_start=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0

        ----------------------------------------------
        Returns:
        result_df : pandas.DataFrame


        Examples
        --------Libs that may be used-----------------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_boston
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import mean_squared_error
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> from sklearn.ensemble import BaggingRegressor
        >>> from sklearn.model_selection import KFold
        >>> from sklearn.model_selection import cross_val_score
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.metrics import make_scorer
        >>> from sklearn.metrics import mean_squared_error
        >>> from sklearn.metrics import r2_score
        >>> import numpy as np
        >>> import warnings
        >>> warnings.filterwarnings('ignore')
        ----------------------------------------------

        """


        result_dict = {}
        result_df = pd.DataFrame(result_dict)
        print("diy_bagging_function")
        return(result_df)
    