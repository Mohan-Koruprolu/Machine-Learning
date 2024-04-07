from .regression import Regression
import numpy as np 

class LinearRegression(Regression):
    # we make our model like this y=x @ W matmul of x and weights
    def fit(self,x_train:np.ndarray,y_train:np.ndarray):
        self.w=np.linalg.pinv(x_train) @ y_train
        self.var = np.mean(np.square(x_train @ self.w - y_train))
    def predict(self,x,return_std: bool=False):
        y=x@self.w
        if return_std:
            # need to see this 
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y,y_std
        return y
    

