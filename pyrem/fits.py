from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

class CustomGPR(RegressorMixin,BaseEstimator):
    
    def __init__(self):
        return None
    
    def fit(self,X,y,sample_weight=None,normalize_y=True,**kwargs):
        """
        Fit GPR by first subtracting a linear fit.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Training data. Prior to fit, they are normalized to zero mean and unit variance.

        y: array_like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight: array_like of shape (n_samples), optional. Default: None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
            If not None:
                (a) the training data for the linear fit are weighted by sample_weights, and
                (b) the inverse of sample_weights is added to the diagonal of the kernel matrix during GPR fitting.

        normalize_y: bool, optional. Default: True.
            Whether or not to normalize the target values y by removing the mean and scaling to unit-variance.
            If True (default), it only applies to the GPR fit.
         """
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
        self._linear_fit(X,y,sample_weight=sample_weight)
        self._gpr_fit(X,y,alpha=1/sample_weight,\
                    normalize_y=normalize_y,\
                    n_restarts_optimizer=0,\
                    **kwargs)
        return None

    def predict(self,X,return_std=False):
        """
        Predict the target corresponding to the query points X.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            The query points.

        return_std: bool, optional. Default: False.
            Whether or not to return the standard deviation of the predictive distribution at the query points X.

        Returns
        -------
        y_mean: array_like of shape (n_samples,) or (n_samples, n_targets)
            Mean of the predictive distribution at the query points X.

        y_std: array_like of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of the predictive distribution at the query points X.
            Only returned when return_std=True.
        """
        if return_std:
            out, std = self.gpr.predict(X,return_std=True)
            out += self.lin.predict(X)
            return out, std
        else:
            out = self.gpr.predict(X,return_std=False)
            out += self.lin.predict(X)
            return out

    def rms_score(self,X,y_true,sample_weight=None):
        """
        Return the root mean squared error of the prediction.

        Prameters
        ---------
        X: array_like of shape (n_samples, n_features)
            The query points.

        y_true: array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight: array_like of shape (n_samples,). Default: None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
            """
        y_pred = self.predict(X,return_std=False)
        out = mean_squared_error(y_true,y_pred,sample_weight=sample_weight,\
                multioutput='uniform_average')**0.5
        return out

    def std_score(self,X,y_true,sample_weight=None):
        """
        Return the root mean squared error of the prediction, where each sample is weighted by the inverse square of the standard deviation of the predcitve distribution.
        It can be thought as the average distance of the mean predictions from y_true, in units of GPR standard deviation.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            The query points.

        y_true: array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight: array_like of shape (n_samples,). Default: None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
        """
        y_pred, std_pred = self.predict(X,return_std=True)
        out = mean_squared_error(y_true,y_pred,sample_weight=sample_weight,\
                multioutput='raw_values')
        out = out/std_pred**2
        out = np.mean(out)**0.5
        return out

    def r2_score(self,X,y_true,sample_weight=None):
        """
        Return the R2 (coefficient of determination) regression score.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            The query points.

        y_true: array_like of shape (n_samples,) or (n_samples, n_targets)
            Correct target values.

        sample_weight: array_like of shape (n_samples,), optional. Default: None.
            Individual weights for each sample.
            None (default) is equivalent to 1-D sample_weight filled with ones.
        """
        y_pred = self.predict(X,return_std=False)
        out = r2_score(y_true,y_pred,sample_weight=sample_weight,\
                   multioutput='uniform_average')
        return out

    def sample_y(self,X,n_samples=1,random_state=42):
        """
        Draw samples from the GPR and evaluate at X.

        Parameters
        ----------
        X: array_like of shape (n_samples_X, n_features)
            The query points.
    
        n_samples: int, optional. Default: 1.
            Number of samples to be drawn per query point.

        Returns
        -------
        y_samples: array-like of shape (n_samples_X, n_samples) or (n_samples_X, n_targets, n_samples)
        """
        X_transformed = self.gpr['scaler'].transform(X)
        out = self.gpr['gpr'].sample_y(X_transformed,n_samples=n_samples,random_state=random_state)
        if len(out.shape)==2:
            out += self.lin.predict(X)[:,np.newaxis]
        elif len(out.shape)==3:
            out += self.lin.predict(X)[:,:,np.newaxis]
        return out

    def _linear_fit(self,X,y,sample_weight=None):
        self.lin = Pipeline([('scaler',StandardScaler()),\
                            ('lin',LinearRegression())],\
                            )
        self.lin.fit(X,y,lin__sample_weight=sample_weight)
        return None

    def _gpr_fit(self,X,y,**kwargs):
        kernel = self._prepare_kernel(X_dim=X.shape[1],\
                   white_kernel=True)
        self.gpr = Pipeline([('scaler',StandardScaler()),\
                              ('gpr',GPR(kernel=kernel,**kwargs))])
        y_lin = self.lin.predict(X)
        self.gpr.fit(X,y-y_lin)
        return None

    def _prepare_kernel(self,X_dim,white_kernel=False):
        kernel1 = RBF(length_scale=np.ones((X_dim,)),length_scale_bounds=(1e-10,1e5))
        kernel2 = ConstantKernel(constant_value_bounds=(1e-5,1e5))
        kernel = kernel1*kernel2
        if white_kernel:
            kernel3 = WhiteKernel(noise_level=1,noise_level_bounds=(1e-10,1e2))
            kernel += kernel3
        return kernel
