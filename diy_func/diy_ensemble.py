import pandas as pd
import numpy as np

class diy_ensemble():
    """
    

    Parameters
    """
    def __init__(self, num_inputs,input_df, n_bootstrap, n_sample, random_state=None):
        self.weights = [0] * num_inputs
        self.bias = 0
        self.input_df = input_df
        self.n_bootstrap = n_bootstrap
        self.n_sample = n_sample
        self.random_state = random_state
        

    def diy_boostrap(self, input_df, n_bootstrap, n_sample, random_state=None):
        """
        This function performs bootstrap sampling on a pandas dataframe.

        ------------------------------
        Parameters
        ----------
        input_df : pandas dataframe
            The input dataframe.

        n_bootstrap : int
            The number of bootstrap samples.

        n_sample : int
            The number of samples in each bootstrap sample.

        random_state : int
            The random seed.

        ------------------------------
        Returns
        -------
        bootstrap_samples : list of DataFrames
            List containing the bootstrap samples.
        """
        if random_state is not None:
         np.random.seed(random_state)

        bootstrap_samples = []
        n_rows = input_df.shape[0]

        for _ in range(n_bootstrap):
            # Generate random indices for the bootstrap sample
            random_indices = np.random.randint(0, n_rows, size=n_sample)
            # Select rows corresponding to the random indices
            sample = input_df.iloc[random_indices]
            bootstrap_samples.append(sample)

        return bootstrap_samples
    
    def diy_bagging(input_df, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        """
        
        
        Parameters
        ----------
        para
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

        ----------------------------------------------
        Bagging(Bootstrap Aggregating) is a ensemble method that combines multiple models to increase accuracy.
        

        Boosting


        Stacking



        ----------------------------------------------

        """


        result_dict = {}
        result_df = pd.DataFrame(result_dict)
        print("diy_bagging_function")
        return(result_df)
    

    def diy_boosting(input_df, n_estimators=10, learning_rate=1.0, loss='linear', random_state=None):
        """
        
        
        Parameters
        ----------
        input_df: pandas.DataFrame
        target_df, 
        n_estimators=10, 
        learning_rate=1.0, 
        loss='linear', 
        random_state=None

        ----------------------------------------------
        Returns:
        result_df : pandas.DataFrame

        ----------------------------------------------
        Bagging(Bootstrap Aggregating) is a ensemble method that combines multiple models to increase accuracy.
        

        Boosting


        Stacking



        ----------------------------------------------

        """


        result_dict = {}
        result_df = pd.DataFrame(result_dict)
        print("diy_boosting_function")
        return(result_df)
    


    def diy_stacking(input_df, n_estimators=10, learning_rate=1.0, loss='linear', random_state=None):
        """
        
        
        Parameters
        ----------
        input_df: pandas.DataFrame
        target_df, 
        n_estimators=10, 
        learning_rate=1.0, 
        loss='linear', 
        random_state=None

        ----------------------------------------------
        Returns:
        result_df : pandas.DataFrame

        ----------------------------------------------
        Stacking



        ----------------------------------------------

        """


        result_dict = {}
        result_df = pd.DataFrame(result_dict)
        print("diy_stacking_function")
        return(result_df)