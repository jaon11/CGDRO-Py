import numpy as np
from utils import *
import cvxpy as cp

class DRoL:
    def __init__(self, f_learner = 'xgb', w_learner = 'linear', seed = 123, verbose = False):
        self.seed = seed
        self.f_learner = f_learner
        self.w_learner = w_learner
        self.verbose = verbose


        self.Gamma_plug = None
        self.Gamma_corr = None
        self.pred_full_mat = None
        self.source_full_models = None
    
    def fit(self, X_list, y_list, X0=None):
        """Compute the plug-in and bias-corrected estimators of the Gamma matrix.

        Args:
            f_learner (str, optional): method used to fit outcome models on each source. Defaults to 'xgb'.
            w_learner (str, optional): method used to fit density models on each source. Defaults to 'xgb'.
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
        # Use the entire source data to fit the source models
        # and predict on the target data
        self.pred_full_mat = np.zeros((N, self.L))
        self.source_full_models = [UtilModels(mode='reg', f_learner=self.f_learner, w_learner=self.w_learner, split=False, seed=self.seed, verbose=self.verbose) for l in range(self.L)]
        for l in range(self.L):
            self.source_full_models[l].fit_f(self.X_list[l], self.y_list[l])
            self.pred_full_mat[:, l] = self.source_full_models[l].model_f.predict(self.X0)
        # The plug-in estimator of Gamma matrix
        self.Gamma_plug = self.pred_full_mat.T @ self.pred_full_mat / N
        



        models = [UtilModels(mode='reg', f_learner=self.f_learner, w_learner=self.w_learner, split=True, seed=self.seed, verbose=self.verbose) for l in range(self.L)]
        indA_list, indB_list = [], []
        for l in range(self.L):
            models[l].fit_f(self.X_list[l], self.y_list[l])
            indA, indB = models[l].indA, models[l].indB
            models[l].fit_w(self.X_list[l], self.X0)
            indA_list.append(indA)
            indB_list.append(indB)


        # ------------ Bias-Corrected Estimator of Gamma matrix ------------
        self.Gamma_corr = self.Gamma_plug.copy()
        
        for k in range(self.L):
            #fk = models[k].pred_f(self.X_list[k], self.X0)[0]
            #fkA, fkB = fk[indA_list[k]], fk[indB_list[k]]
            #wk = models[k].pred_w(self.X_list[k], self.X0)
            #wkA, wkB = wk[indA_list[k]], wk[indB_list[k]]
            fkA = models[k].modelA_f
            fkB = models[k].modelB_f
            wkA = models[k].modelA_w
            wkB = models[k].modelB_w

           
            
            for l in range(self.L):
                flA = models[l].modelA_f
                flB = models[l].modelB_f
                wlA = models[l].modelA_w
                wlB = models[l].modelB_w
                #fl = models[l].pred_f(self.X_list[l], self.X0)[0]
                #flA, flB = fl[indA_list[l]], fl[indB_list[l]]
                #wl = models[l].pred_w(self.X_list[l], self.X0)
                #wlA, wlB = wl[indA_list[l]], wl[indB_list[l]]
                #fkl = models[k].pred_f(self.X_list[l], self.X0)[0]
                #fklA, fklB = fkl[indA_list[l]], fkl[indB_list[l]]
                #flk = models[l].pred_f(self.X_list[k], self.X0)[0]
                #flkA, flkB = flk[indA_list[k]], flk[indB_list[k]]
   

                #num1A = np.mean(wlA * fklA * (flA - self.y_list[l][indA_list[l]]))
                #num2A = np.mean(wkA * flkA * (fkA - self.y_list[k][indA_list[k]]))
                #num1B = np.mean(wlB * fklB * (flB - self.y_list[l][indB_list[l]]))
                #num2B = np.mean(wkB * flkB * (fkB - self.y_list[k][indB_list[k]]))

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
            Yl : label array on the l-th source domain
        """
        return np.mean(wl.predict(Xl) * fk.predict(Xl) * (fl.predict(Xl) - Yl))

    

        