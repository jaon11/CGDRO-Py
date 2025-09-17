import numpy as np
from utils import *
import cvxpy as cp
from cvxpy.error import DCPError, SolverError

class Regression:
    """
    Distributionally Robust Learning (DRoL) for multi-source data.
        Args:
            f_learner (str, optional): method used to fit outcome models on each source. Defaults to 'xgb'.
            w_learner (str, optional): method used to fit density models on each source. Defaults to 'xgb'.
            seed (int, optional): random seed. Defaults to 123.
            verbose (bool, optional): whether to print out the fitting information. Defaults to False.
    """
    def __init__(self, f_learner = 'xgb', w_learner = 'linear', seed = 123, verbose = False):
        self.seed = seed
        self.f_learner = f_learner
        self.w_learner = w_learner
        self.verbose = verbose


        self.Gamma_plug = None
        self.Gamma_corr = None
        self.pred_full_mat = None
        self.source_full_models = None

    def fit(self, X_list, y_list, X0=None, bias_correct=True, priors=None):
        """Compute the plug-in and bias-corrected estimators of the Gamma matrix.

        Args:
            X_list (list): list of feature matrices on each source domain
            y_list (list): list of label arrays on each source domain
            X0 (array, optional): feature matrix on the target domain. If None, use
                                    the pooled source data as the target data. Defaults to None.
            bias_correct (bool, optional): whether to use the bias-corrected estimator of the Gamma matrix. Defaults to True.
            priors (tuple, optional): prior information on the aggregation weights, given as (prior_weight, rho),
                                    where prior_weight is the prior weight vector and rho is the radius of the
                                    L2-norm ball around prior_weight. If None, no prior information is used. Defaults to None.
        """
        self.X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]
        self.y_list = [np.asarray(yi, dtype=float).ravel() for yi in y_list]
        self.L = len(self.X_list)  # Number of source domains
        self.d = self.X_list[0].shape[1]   # Feature dimension
        nl_s = [self.X_list[l].shape[0] for l in range(self.L)]
        if X0 is None:
            X0 = np.vstack(self.X_list)
        else:
            X0 = X0
        X0 = np.asarray(X0, dtype=float)
        self.X0 = X0
        N = self.X0.shape[0]

        # ------------ Plug-in Estimator of Gamma matrix ------------
        self.pred_full_mat = np.zeros((N, self.L))
        self.source_full_models = [UtilModels(mode='reg', f_learner=self.f_learner, w_learner=self.w_learner, split=False, seed=self.seed, verbose=self.verbose) for l in range(self.L)]
        for l in range(self.L):
            self.source_full_models[l].fit_f(self.X_list[l], self.y_list[l])
            self.pred_full_mat[:, l] = self.source_full_models[l].model_f.predict(self.X0)
        # The plug-in estimator of Gamma matrix
        self.Gamma_plug = self.pred_full_mat.T @ self.pred_full_mat / N
        



        models = [UtilModels(mode='reg', f_learner=self.f_learner, w_learner=self.w_learner, split=True, seed=self.seed, verbose=self.verbose) for l in range(self.L)]
        indA_list, indB_list = [], []
        w_list = []
        for l in range(self.L):
            models[l].fit_f(self.X_list[l], self.y_list[l])
            indA, indB = models[l].indA, models[l].indB
            models[l].fit_w(self.X_list[l], self.X0)
            w_list.append(models[l].pred_w(self.X_list[l], self.X0))
            indA_list.append(indA)
            indB_list.append(indB)


        # ------------ Bias-Corrected Estimator of Gamma matrix ------------
        self.Gamma_corr = self.Gamma_plug.copy()
        
        for k in range(self.L):
            fkA = models[k].modelA_f
            fkB = models[k].modelB_f
            wkA = w_list[k][indB_list[k]]
            wkB = w_list[k][indA_list[k]]

           
            
            for l in range(self.L):
                flA = models[l].modelA_f
                flB = models[l].modelB_f
                wlA = w_list[l][indB_list[l]]
                wlB = w_list[l][indA_list[l]]

                num1A = self._bias_correct(fkA, flA, wlA,
                                           self.X_list[l][indB_list[l]],
                                           self.y_list[l][indB_list[l]])
                num2A = self._bias_correct(flA, fkA, wkA,
                                           self.X_list[k][indB_list[k]],
                                           self.y_list[k][indB_list[k]])
                num1B = self._bias_correct(fkB, flB, wlB,
                                           self.X_list[l][indA_list[l]],
                                           self.y_list[l][indA_list[l]])
                num2B = self._bias_correct(flB, fkB, wkB,
                                           self.X_list[k][indA_list[k]],
                                           self.y_list[k][indA_list[k]])

                self.Gamma_corr[k, l] -= (num1A + num2A + num1B + num2B) / 2
        
        self.Gamma_corr = (self.Gamma_corr.T + self.Gamma_corr) / 2
        Gamma = self.Gamma_corr if bias_correct else self.Gamma_plug
        q = cp.Variable(self.L, nonneg=True)
        if priors is None:
            constraints = [cp.sum(q) == 1]
        else:
            prior_weight, rho = priors
            constraints = [cp.sum(q) == 1, cp.norm(q - prior_weight) <= rho]
        objective = cp.Minimize(cp.quad_form(q, Gamma))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve()
            q_opt = q.value
            self.weight_ = q_opt
        except (DCPError, SolverError) as e:
            print("f_learner or w_learner is not well learned.")
            self.weight_ = None
        
    def predict(self):
        """Estimate the optimal aggregation weights using the estimated Gamma matrix,
        and yield the robust prediction on the target domain.
        
        Returns:
            pred : the robust prediction on the target domain
        """

        pred = self.pred_full_mat @ self.weight_
        self.pred = pred

        return pred

    def _bias_correct(self, fk, fl, wl, Xl, Yl):
        """Compute the bias corrected term: mean[wl * fk(Xl) * (fl(Xl) - Yl)],
        where the models fk, fl, wl are independent of the data Xl, Yl.

        Args:
            fk (Instance of OutComeModel): fitted outcome model on the k-th source domain
            fl (Instance of OutComeModel): fitted outcome model on the l-th source domain
            wl (Instance of DensityModel): fitted density ratio model on the l-th source domain
            Xl : feature matrix on the l-th source domain
            Yl : label array on the l-th source domain
        """
        
        return np.mean(wl * fk.predict(Xl) * (fl.predict(Xl) - Yl))

    

        