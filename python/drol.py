import numpy as np
from utils import *
import cvxpy as cp

class DRoL:
    def __init__(self, outcome_learner='xgb', density_learner='xgb', intercept=False):
        self.outcome_learner = outcome_learner
        self.density_learner = density_learner
        self.intercept = intercept
        self.Gamma_plug = None
        self.Gamma_corr = None
        self.pred_full_mat = None
        self.source_full_models = None
    
    def fit(self, X_list, y_list, X0=None):
        """Compute the plug-in and bias-corrected estimators of the Gamma matrix.

        Args:
            outcome_learner (str, optional): method used to fit outcome models on each source. Defaults to 'xgb'.
            density_learner (str, optional): method used to fit density models on each source. Defaults to 'xgb'.
        """
        self.X_list = [np.asarray(Xi - np.mean(Xi, axis=0), dtype=float) for Xi in X_list]
        self.y_list = [np.asarray(yi, dtype=float).ravel() for yi in y_list]
        self.L = len(self.X_list)  # Number of source domains
        self.d = self.X_list[0].shape[1] + (1 if self.intercept else 0)  # Feature dimension
        nl_s = [self.X_list[l].shape[0] for l in range(self.L)]
        if X0 is None:
            X0 = np.vstack(self.X_list)
        else:
            X0 = X0
        X0 = np.asarray(X0, dtype=float)
        X0 = X0 - np.mean(X0, axis=0)
        self.X0 = X0
        N = self.X0.shape[0]

        if self.intercept:
            self.X0 = np.hstack([np.ones((self.X0.shape[0], 1), dtype=float), self.X0])
            self.X_list = [np.hstack([np.ones((Xi.shape[0], 1), dtype=float), Xi]) for Xi in self.X_list]

        # ------------ Plug-in Estimator of Gamma matrix ------------
        # Use the entire source data to fit the source models
        # and predict on the target data
        self.pred_full_mat = np.zeros((N, self.L))
        self.source_full_models = [OutcomeModel(learner=self.outcome_learner, params=None) for l in range(self.L)]
        for l in range(self.L):
            self.source_full_models[l].fit(self.X_list[l], self.y_list[l])
            self.pred_full_mat[:, l] = self.source_full_models[l].predict(self.X0)
        # The plug-in estimator of Gamma matrix
        self.Gamma_plug = self.pred_full_mat.T @ self.pred_full_mat / N
        
        # Use the sample-split source data to fit
        source_A_models = [OutcomeModel(learner=self.outcome_learner, params=None) for l in range(self.L)]
        source_B_models = [OutcomeModel(learner=self.outcome_learner, params=None) for l in range(self.L)]
        density_A_models = [DensityModel(learner=self.density_learner, params=None) for l in range(self.L)]
        density_B_models = [DensityModel(learner=self.density_learner, params=None) for l in range(self.L)]
        for l in range(self.L):
            half_l = nl_s[l] // 2
            source_A_models[l].fit(self.X_list[l][:half_l], self.y_list[l][:half_l])
            source_B_models[l].fit(self.X_list[l][half_l:], self.y_list[l][half_l:])
            density_A_models[l].fit(self.X_list[l][:half_l], self.X0)
            density_B_models[l].fit(self.X_list[l][half_l:], self.X0)
        
        # ------------ Bias-Corrected Estimator of Gamma matrix ------------
        self.Gamma_corr = self.Gamma_plug.copy()
        
        for k in range(self.L):
            fkA = source_A_models[k]
            fkB = source_B_models[k]
            wkA = density_A_models[k]
            wkB = density_B_models[k]
            half_k = nl_s[k] // 2
            
            for l in range(self.L):
                flA = source_A_models[l]
                flB = source_B_models[l]
                wlA = density_A_models[l]
                wlB = density_B_models[l]
                half_l = nl_s[l] // 2
                
                num1A = self._bias_correct(fkA, flA, wlA,
                                           self.X_list[l][half_l:],
                                           self.y_list[l][half_l:])
                num2A = self._bias_correct(flA, fkA, wkA,
                                           self.X_list[k][half_k:],
                                           self.y_list[k][half_k:])
                num1B = self._bias_correct(fkB, flB, wlB,
                                           self.X_list[l][:half_l],
                                           self.y_list[l][:half_l])
                num2B = self._bias_correct(flB, fkB, wkB,
                                           self.X_list[k][:half_k],
                                           self.y_list[k][:half_k])
                self.Gamma_corr[k, l] -= (num1A + num2A + num1B + num2B) / 2
        
        self.Gamma_corr = (self.Gamma_corr.T + self.Gamma_corr) / 2
        
    def predict(self, bias_correct=True, priors=None):
        """Estimate the optimal aggregation weights using the estimated Gamma matrix,
        and yield the robust prediction on the target domain.

        Args:
            bias_correct (bool, optional): whether to use the bias-corrected estimator. Defaults to True.
            priors (list, optional): priors upon the aggregation weights. Defaults to None.
        
        Returns:
            pred : the robust prediction on the target domain
            q_opt : the optimal aggregation weights
        """
        Gamma = self.Gamma_corr if bias_correct else self.Gamma_plug
        q = cp.Variable(self.L, nonneg=True)
        if priors is None:
            constraints = [cp.sum(q) == 1]
        else:
            prior_weight, rho = priors
            constraints = [cp.sum(q) == 1, cp.norm(q - prior_weight) <= rho]
        objective = cp.Minimize(cp.quad_form(q, Gamma))
        prob = cp.Problem(objective, constraints)
        prob.solve()
        q_opt = q.value
        pred = self.pred_full_mat @ q_opt
        self.weight_ = q_opt
        self.pred = pred
        result = {
            'weight_': self.weight_,
            'pred': self.pred
        }
        return result
    
    def _bias_correct(self, fk, fl, wl, Xl, Yl):
        """Compute the bias corrected term: mean[wl * fk(Xl) * (fl(Xl) - Yl)],
        where the models fk, fl, wl are independent of the data Xl, Yl.

        Args:
            fk (Instance of OutComeModel): fitted outcome model on the k-th source domain
            fl (Instance of OutComeModel): fitted outcome model on the l-th source domain
            wl (Instance of DensityModel): fitted density ratio model on the l-th source domain
            Xl : feature matrix on the l-th source domain
            Yl : lable array on the l-th source domain
        """
        return np.mean(wl.predict(Xl) * fk.predict(Xl) * (fl.predict(Xl) - Yl))
    


    ### intercept: take or not?
        