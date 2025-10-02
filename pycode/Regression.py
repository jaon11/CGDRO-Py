import numpy as np
from utils import *
import cvxpy as cp
from cvxpy.error import DCPError, SolverError
from scipy.stats import norm, chi2
from scipy.linalg import svd


class linear:
    class ld:
        """
        Closed-form DRO linear regression (low-dimensional case)

        Args:
            f_learner (str, optional): method used to fit outcome models on each source
            intercept (bool, optional): whether to include intercept in outcome models. Defaults to False.
            loading_intercept (bool, optional): whether to include intercept in loading matrix. Defaults to False.
            delta (float, optional): ridge penalty level, non-positive. Defaults to 0.
            lam (float, optional): Lasso penalty level for high-dimensional regression. Defaults to None.
            verbose (bool, optional): whether to print out the fitting information. Defaults to False.
        """
        def __init__(self, intercept=False,delta=0, verbose=False):
            self.intercept = intercept
            self.verbose = verbose
            self.delta = delta

    # ==================================================================================================== #
    # =================== Run Closed-form solution to solve gamma and theta ============================== #
    # ==================================================================================================== #
        def fit(self, X_list, y_list, X0=None, loss_type='reward'):
            """
            Fit (point estimate) the linear regression model using closed-form DRO.

            Args:
                X_list (list of array-like): list of source domain features, each element is n_i x d.
                y_list (list of array-like): list of source domain labels, each element is n_i x 1.
                loss_type (str, optional): type of the loss function used to compute the optimal aggregation weights.
                                        Options include 'reward' (default), 'squaredloss', and 'regret'. Defaults to 'reward'.
                X0 (array-like, optional): target domain features, n0 x d. If None, use all sources' data. Defaults to None.
            """

            self.X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]
            self.y_list = [np.asarray(yi, dtype=float).ravel() for yi in y_list]
            self.L = len(self.X_list)  # Number of source domains
            self.d = self.X_list[0].shape[1]  + (1 if self.intercept else 0)  # Feature dimension




            if not isinstance(self.verbose, bool):
                self.verbose = True
            

            if self.verbose:
                print('start fitting-----')
            ### Fitting Bias-corrected Estimator of Coef_ with loading matrix ###


            ### Fitting OLS of each group ###
            ## coef_ of each group
            beta_list = []
            for l in range(self.L):
                X_l = X_list[l] - np.mean(X_list[l], axis=0)
                y_l = y_list[l]
                if self.intercept:
                    X_l = np.hstack([np.ones((X_l.shape[0], 1), dtype=float), X_l])
                model = LinearRegression().fit(X_l, y_l)
                beta_list.append(model.coef_)
            self.beta_list = np.array(beta_list)  # Shape: (L, d)
            ## Gamma
            if X0 is None:
                X0 = np.vstack(self.X_list)  # Target domain features
            else:
                X0 = X0          # Target domain features
            X0 = np.asarray(X0, dtype=float)
            X0 = X0 - np.mean(X0, axis=0)
            self.X0 = X0

            if self.intercept:
                X0_with_int = np.column_stack((np.ones(X0.shape[0]), X0))
                Sigma0 = (X0_with_int.T @ X0_with_int) / X0_with_int.shape[0]
            else:
                Sigma0 = (X0.T @ X0) / X0.shape[0] # Shape: (d, d)
            Gamma = np.zeros((self.L, self.L))  # Two-dimensional array
            for l in range(self.L):
                for k in range(self.L):
                    Gamma[l, k] = self.beta_list[l] @ Sigma0 @ self.beta_list[k]
            self.Gamma = Gamma
            self.mean_Gamma = self.Gamma[np.tril_indices(self.L)]

            ### sd of beta_list ###
            dev_vec = np.zeros(self.L)
            var_mat = np.zeros((self.L,self.d))  # Shape: (L, d)
            for l in range(self.L):
                X_l = self.X_list[l]
                y_l = self.y_list[l]
                dev_vec[l] = np.sum((y_l - X_l @ self.beta_list[l]) ** 2) / (X_l.shape[0] - self.d)
                varl = dev_vec[l] * np.linalg.inv(X_l.T @ X_l)  # Shape: (n_loading, n_loading)
                var_mat[l, :] = np.diag(varl)  # Store diagonal elements
            
            self.dev_vec = dev_vec
            self.var_mat = var_mat

            ### Fitting DRO regression ###
            ## optimized weight vector
            self.loss_type = loss_type
            self.weight_ = self.opt_weight(self.Gamma,loss_type=self.loss_type)['weight']
            ## DRO regression coefficients
            self.coef_ = self.beta_list.T @ self.weight_  # Shape: (d,)



            self.parameters = {
                    'coef_': self.coef_,
                    'weight_': self.weight_
                }



    # ======================================================================= #
    # =================== Prediction  ======================================= #
    # ======================================================================= #
        def predict(self):
            """
            Predict using the fitted DRO regression model.

            Returns:
                pred (array-like): Predicted values for the target domain, shape (n0,).
            """

            pred = self.X0 @ self.coef_

            return pred

    # ======================================================================= #
    # =================== Compute CIs ======================================= #
    # ======================================================================= #
        def infer(self, M=200, alpha=0.05, alpha_thres=0.01):        
            """
            Perform resampling-based inference to compute confidence intervals for the loading coefficients.
            
            Args:
                M (int, optional): Number of resampling iterations. Defaults to 500.
                alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.
                alpha_thres (float, optional): Threshold for generating samples. Defaults to 0.01.  
            """
            if not hasattr(self, 'coef_'):
                raise ValueError("Model is not fitted yet. Please call 'fit' first.")
            
            if self.loss_type != 'reward':
                raise ValueError("Currently only support inference for loss_type='reward'.")

            ### Sampling ###
            Var_Gamma = self.compute_Var_Gamma(tau=0.2)
            self.mean_Gamma = self.mean_Gamma.reshape(-1, 1)  # Ensure mu is a column vector
            gen_samples = self.gensamples(Var_Gamma, gen_size=M, threshold=0, alpha_thres=alpha_thres) # Shape: (M, gen_dim)

            gen_weight_mat = np.empty((M, self.L))

            for g in range(M):
                gen_matrix = np.full((self.L, self.L), np.nan)

                # Fill lower triangle and diagonal
                tril_indices = np.tril_indices(self.L)
                gen_matrix[tril_indices] = gen_samples[g, :]

                # Fill upper triangle by symmetry
                gen_matrix = gen_matrix + np.triu(gen_matrix.T, k=1)

                # Solve for optimal weights
                gen_sol = self.opt_weight(self.Gamma, loss_type='reward')
                gen_weight_mat[g, :] = gen_sol["weight"]

            ### Constructing CIs ###
            CIs = np.zeros((self.d, 2))
            for k in range(self.d):
                gen_coef_ = (gen_weight_mat @ self.beta_list[:,k]).reshape(-1)  # Shape: (M,)
                ses = np.sqrt(self.var_mat[:, k])  # Standard errors for each source domain, shape (L,)
                gen_se = (gen_weight_mat @ ses).reshape(-1)  # Shape: (M,)
                    # Compute confidence intervals
                z_alpha = norm.ppf(1 - alpha / 2)
                gen_CIs_lb = gen_coef_ - z_alpha * gen_se # Shape: (M,)
                gen_CIs_ub = gen_coef_ + z_alpha * gen_se # Shape: (M,)
                CIs[k, 0] = np.min(gen_CIs_lb)
                CIs[k, 1] = np.max(gen_CIs_ub)  
            self.CI = CIs


    # ======================================================================= #
    # =================== Summary Functions ================================= #
    # ======================================================================= #
        def summary(self, index=None):
            """
            Print a summary of the fitted model, including coefficients, weights, and CIs.

            Args:
                index (list or int, optional): Specific dimensions to display. Defaults to None (all dimensions).
            """
            if not hasattr(self, 'parameters'):
                raise ValueError("Model is not fitted yet. Please call 'fit' first.")

            # ---- helpers ----
            def _normalize_indices(user_idx, lo_1based, hi_1based, name):
                if user_idx is None:
                    return list(range(lo_1based - 1, hi_1based))  # all valid 0-based
                try:
                    idx_list = list(user_idx)
                except TypeError:
                    idx_list = [user_idx]
                norm = []
                for v in idx_list:
                    vv = int(v)
                    if not (lo_1based <= vv <= hi_1based):
                        raise ValueError(f"{name} out of range: {vv} not in [{lo_1based}, {hi_1based}]")
                    norm.append(vv - 1)  # to 0-based
                # de-dup preserve order
                seen, dedup = set(), []
                for x in norm:
                    if x not in seen:
                        seen.add(x)
                        dedup.append(x)
                return dedup

            def _print_chunks(label, indices, values, width=8, per_row=10, fmt="{:>8.4f}", header_label="index"):
                """Pretty-print header+row in chunks."""
                # values is assumed aligned to indices order
                for start in range(0, len(indices), per_row):
                    chunk_idx  = indices[start:start+per_row]
                    chunk_vals = values[start:start+per_row]
                    header = f"{header_label:<10}| " + " ".join(f"{(i+1):>{width}}" for i in chunk_idx)
                    row    = f"{label:<10}| " + " ".join(fmt.format(v) for v in chunk_vals)
                    print(header)
                    print(row)

            # ---- data prep ----
            coef = self.parameters['coef_']   # shape (n_loading,)
            d = len(coef)
            dim_idx = _normalize_indices(index, 1, d, "index")

            print("Model Summary:")
            print("=================================")

            # ---- Weights (as table: group | 1..L) ----
            weight = self.parameters['weight_']  # shape (L,)
            L = len(weight)
            print("CGDRO Aggregated Weights:\n")
            group_idx = list(range(L))
            _print_chunks("weight_", group_idx, list(weight), width=8, per_row=10, fmt="{:>8.4f}", header_label="group")
            print()

            print("=================================")

            # ---- Plug-in estimates ----
            print("Coefficient Estimators:\n")
            plug_vals = [coef[i] for i in dim_idx]
            _print_chunks("coef_", dim_idx, plug_vals, width=8, per_row=10, fmt="{:>8.4f}", header_label="index")
            print()

            # ---- Confidence Intervals ----
            if hasattr(self, 'CI'):
                CI = self.CI  # shape (n_loading, 2)
                print("=================================")
                print("Confidence Intervals:\n")

                # Pre-format as tuple strings; 5 per row for readability
                ci_strs = [f"({CI[i,0]:.4f},{CI[i,1]:.4f})" for i in dim_idx]
                _print_chunks("CI", dim_idx, ci_strs, width=14, per_row=5, fmt="{}", header_label="index")
                print()
            else:
                print("Confidence Intervals not computed. Please run infer() method.")

    # ========================================================== #
    # =================== Internal Functions =================== #
    # ========================================================== #

        def opt_weight(self, Gamma, loss_type='reward'):
            """
            Compute Ridge-type weight vector using cvxpy.
            
            Args:
                Gamma (array-like): L x L matrix.
                loss_type (str, optional): Type of loss function ('reward', 'squaredloss or 'regret'). Defaults to 'reward'.
            Returns:
                dict: Contains 'weight' (optimal weights) and optionally 'reward' (optimal reward value).
            """
            Gamma = np.array(Gamma, dtype=float)
            Gamma_diag = cp.diag(Gamma)
            L = Gamma.shape[1]
            
            # Eigen-decomposition and positive adjustment
            eigvals, eigvecs = np.linalg.eigh(Gamma)
            eigvals = np.maximum(eigvals, 0.001)  # ensure positive definite
            Gamma_positive = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Define variable
            v = cp.Variable(L, nonneg=True)
            if loss_type == 'reward':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive + np.eye(L) * self.delta))
            elif loss_type == 'squaredloss':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive + np.eye(L) * self.delta) - cp.matmul(v, Gamma_diag + self.dev_vec))
            elif loss_type == 'regret':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive + np.eye(L) * self.delta) - cp.matmul(v, Gamma_diag))
            constraints = [cp.sum(v) == 1]
            prob = cp.Problem(objective, constraints)
            
            prob.solve()
            opt_weight = v.value
            opt_weight[np.abs(opt_weight) <= 1e-8] = 0.0  # threshold
            
            return {"weight": opt_weight}


        def index_map(self, l, k):
            """
            Maps (l, k) with l >= k into vectorized index of upper-triangular part (column-wise),
            matching R: index.map(L, l, k) = (2L - k)(k - 1)/2 + l

            Args:
                l (int): row index (0-based)
                k (int): column index (0-based), must satisfy l >= k
            Returns:
                int: vectorized index (0-based)
            """
            return int((2 * self.L - (k + 1)) * k // 2 + l)

        def compute_Var_Gamma(self, tau=0.2):
            L = len(self.X_list)
            ns = np.array([X.shape[0] for X in self.X_list])  # Number of samples in each source domain
            gen_dim = L * (L + 1) // 2
            Var_Gamma = np.full((gen_dim, gen_dim), np.nan)

            for k1 in range(L):
                for l1 in range(k1, L):
                    for k2 in range(L):
                        for l2 in range(k2, L):
                            ind1 = self.index_map(l1, k1)
                            ind2 = self.index_map(l2, k2)

                            X_l1 = self.X_list[l1].copy()
                            X_k1 = self.X_list[k1].copy()

                            if self.intercept:
                                X_l1 = np.hstack((np.ones((X_l1.shape[0], 1)), X_l1))
                                X_k1 = np.hstack((np.ones((X_k1.shape[0], 1)), X_k1))

                            Sigma_l1 = X_l1.T @ X_l1 / X_l1.shape[0]
                            Sigma_k1 = X_k1.T @ X_k1 / X_k1.shape[0]

                            dev_l1 = sum((self.y_list[l1] - X_l1 @ self.beta_list[l1]) ** 2)/ (X_l1.shape[0]-self.d)
                            dev_k1 = sum((self.y_list[k1] - X_k1 @ self.beta_list[k1]) ** 2)/ (X_k1.shape[0]-self.d)

                            # projection vectors
                            Proj1 = np.linalg.inv(np.cov(X_l1, rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[k1]
                            Proj2_l1 = np.linalg.inv(np.cov(self.X_list[l2], rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[k2] if l2 == l1 else np.zeros(self.d)
                            Proj2_k1 = np.linalg.inv(np.cov(self.X_list[k2], rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[l2] if k2 == l1 else np.zeros(self.d)
                            val1 = dev_l1 / X_l1.shape[0] * (Proj1 @ Sigma_l1 @ (Proj2_l1 + Proj2_k1))

                            Proj3 = np.linalg.inv(np.cov(X_k1, rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[l1]
                            Proj4_l1 = np.linalg.inv(np.cov(self.X_list[l2], rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[k2] if l2 == k1 else np.zeros(self.d)
                            Proj4_k1 = np.linalg.inv(np.cov(self.X_list[k2], rowvar=False)) @ np.cov(self.X0, rowvar=False) @ self.beta_list[l2] if k2 == k1 else np.zeros(self.d)
                            val2 = dev_k1 / X_k1.shape[0] * (Proj3 @ Sigma_k1 @ (Proj4_l1 + Proj4_k1))

                            P1 = self.X0 @ self.beta_list[l1] * (self.X0 @ self.beta_list[k1])
                            P2 = self.X0 @ self.beta_list[l2] * (self.X0 @ self.beta_list[k2])
                            val3 = np.mean((P1 - np.mean(P1)) * (P2 - np.mean(P2))) / self.X0.shape[0]
                        
                        
                            val = val1 + val2 + val3
                            Var_Gamma[ind1, ind2] = val

            # Regularize the diagonal
            diag_correction = np.maximum(tau * np.diag(Var_Gamma), 1.0 / np.min(ns))
            Var_Gamma += np.diag(diag_correction)

            return Var_Gamma


        def gensamples(self, gen_Cov, gen_size=500, threshold=0, alpha_thres=0.01):
            """
            Generate samples from a multivariate normal distribution with different truncation strategies.

            Args:
                gen_Cov (array-like): Covariance matrix for the multivariate normal distribution.
                gen_size (int, optional): Number of samples to generate. Defaults to 500.   
                threshold (int, optional): Truncation strategy (0: coordinate-wise, 1: chi-square, 2: none). Defaults to 0.
                alpha_thres (float, optional): Significance level for truncation. Defaults to 0.01.
            Returns:
                array-like: Generated samples of shape (gen_size, gen_dim).
            """
            gen_mu = np.asarray(self.mean_Gamma).reshape(-1)  # Ensure gen_mu is a 1D array
            gen_dim = len(gen_mu)
            gen_Cov = np.asarray(gen_Cov)

            # Output container
            gen_samples = np.zeros((gen_size, gen_dim))
            n_picked = 0

            # Threshold 0: coordinate-wise truncation based on Normal quantile
            if threshold == 0:
                thres = norm.ppf(1 - alpha_thres / (gen_dim * 2))
                while n_picked < gen_size:
                    S = np.random.multivariate_normal(mean=np.zeros(gen_dim), cov=gen_Cov)
                    if np.max(np.abs(S / np.sqrt(np.diag(gen_Cov)))) <= thres:
                        gen_samples[n_picked, :] = gen_mu + S
                        n_picked += 1

            # Threshold 1: chi-square truncation
            elif threshold == 1:
                U, D, Vt = svd(gen_Cov)
                D_sqrt = np.sqrt(D)
                gen_Cov_sqrt = U @ np.diag(D_sqrt) @ Vt
                thres = chi2.ppf(1 - alpha_thres, df=gen_dim)
                while n_picked < gen_size:
                    Z = np.random.normal(size=gen_dim)
                    Z_normsq = np.sum(Z**2)
                    if Z_normsq <= thres:
                        gen_samples[n_picked, :] = gen_mu + gen_Cov_sqrt @ Z
                        n_picked += 1

            # Threshold 2: no threshold (standard MVN sampling)
            elif threshold == 2:
                gen_samples = np.random.multivariate_normal(mean=gen_mu, cov=gen_Cov, size=gen_size)

            else:
                raise ValueError("threshold must be 0, 1, or 2.")

            return gen_samples


        


        def dev_fun(self, pred, y, sparsity=0):
            pred = np.asarray(pred).ravel()  
            y = np.asarray(y).ravel()
            n = len(y)
            sigmasq_hat = np.sum((y - pred) ** 2) / max(0.7 * n, n - sparsity)
            return sigmasq_hat



    class hd:
        """
        Closed-form DRO linear regression (high-dimensional case)

        Args:
            f_learner (str, optional): method used to fit outcome models on each source
            intercept (bool, optional): whether to include intercept in outcome models. Defaults to False.
            loading_intercept (bool, optional): whether to include intercept in loading matrix. Defaults to False.
            delta (float, optional): ridge penalty level, non-positive. Defaults to 0.
            lam (float, optional): Lasso penalty level for high-dimensional regression. Defaults to None.
            verbose (bool, optional): whether to print out the fitting information. Defaults to False.
        """
        def __init__(self,intercept=False, loading_intercept=False, delta=0, lam=None, verbose=False):
            self.intercept = intercept
            self.loading_intercept = loading_intercept
            self.lam = lam
            self.verbose = verbose
            self.delta = delta

    # ==================================================================================================== #
    # =================== Run Closed-form solution to solve gamma and theta ============================== #
    # ==================================================================================================== #
        def fit(self, X_list, y_list, index, X0=None):
            """
            Fit (point estimate) the linear regression model using closed-form DRO.

            Args:
                X_list (list of array-like): list of source domain features, each element is n_i x d.
                y_list (list of array-like): list of source domain labels, each element is n_i x 1.
                index (int): index of the loading vector (1-based), the index-th coefficient is of interest.
                X0 (array-like, optional): target domain features, n0 x d. If None, use all sources' data. Defaults to None.
            """

            self.X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]
            self.y_list = [np.asarray(yi, dtype=float).ravel() for yi in y_list]
            
            self.L = len(self.X_list)  # Number of source domains
            self.d = self.X_list[0].shape[1]  + (1 if self.intercept else 0)  # Feature dimension
            bc = BiasCorrection(
                lam=self.lam,                         # if you have these attrs; otherwise omit
                intercept=self.intercept,
                loading_intercept=self.loading_intercept,
                verbose=self.verbose
            )

            index = [i - 1 for i in index] if isinstance(index, (list, tuple)) else index - 1  # Convert to 0-based
            self.index = index
            try:
                n_index = len(index)
            except TypeError:
                index = [index]
                n_index = len(index)
            if any((i < 0 or i >= self.d) for i in index):
                raise ValueError(f"index must be between 0 and {self.d - 1}")
            
            loading_mat = np.zeros((n_index, self.d))
            for i in range(n_index):
                loading_mat[i, index[i]] = 1
            self.loading_mat = np.asarray(loading_mat, dtype=float) # Shape: (n_index, d) or (n_index, d+1) if loading_intercept is True

            if not isinstance(self.verbose, bool):
                self.verbose = True
            if (not self.intercept) and self.loading_intercept:
                self.loading_intercept = False
            if self.verbose:
                print("Argument 'loading_intercept' set to False because intercept is False")



            if self.verbose:
                print('start fitting-----')
            ### Fitting Bias-corrected Estimator of Coef_ with loading matrix ###
            if self.verbose:
                print("======> Bias Correction for initial estimators....")

            init_est = [None] * self.L
            debias_est = [None] * self.L

            for l in range(self.L):
                # center each source X (no scaling)
                self.X_list[l] = self.X_list[l] - np.mean(self.X_list[l], axis=0)
                y = self.y_list[l]
                X = self.X_list[l]
                UM = UtilModels(mode='reg', f_learner='high_d', lambda_val=self.lam, split=False, verbose=self.verbose)
                UM.fit_f(X, y)
                Umodel = UM.model_f
                beta_init = np.concatenate(([Umodel.intercept_], Umodel.coef_)) if self.intercept else Umodel.coef_
                beta_init = np.asarray(beta_init).ravel()
                sparsity = np.sum(np.abs(beta_init) > 1e-4)
                pred = (X @ beta_init).ravel()
                dev = self.dev_fun(pred, y, sparsity=sparsity)


                
                Est = bc.LF(X, y, self.loading_mat, beta_init=beta_init)

                init_est[l] = {'beta_init': beta_init, 'dev': dev}
                debias_est[l] = {'est_debias_vec': np.asarray(Est['est_debias_vec']),
                                'se_vec': np.asarray(Est['se_vec'])}  
            self.init_est = init_est
            self.debias_est = debias_est

            ### Fitting Bias-corrected Estimator of Gamma with loading matrix ###
            if X0 is None:
                X0 = np.vstack(self.X_list)  # Target domain features
            else:
                X0 = X0          # Target domain features
            X0 = np.asarray(X0, dtype=float)
            X0 = X0 - np.mean(X0, axis=0)
            self.X0 = X0
            # pred0.mat: n0 x L
            pred0_mat = np.empty((X0.shape[0], self.L))
            for l in range(self.L):
                pred0_mat[:, l] = (X0 @ self.init_est[l]['beta_init']).ravel()
            self.pred0_mat = pred0_mat

            if self.intercept:
                X0_with_int = np.column_stack((np.ones(X0.shape[0]), X0))
                Sigma0 = (X0_with_int.T @ X0_with_int) / X0_with_int.shape[0]
            else:
                Sigma0 = (X0.T @ X0) / X0.shape[0]

            # Gamma.plugin
            Gamma_plugin = np.zeros((self.L, self.L))
            for l in range(self.L):
                for k in range(l, self.L):
                    b_l = init_est[l]['beta_init']
                    b_k = init_est[k]['beta_init']
                    Gamma_plugin[l, k] = float(b_l.T @ Sigma0 @ b_k)
            # fill symmetric
            for l in range(1, self.L):
                for k in range(0, l):
                    Gamma_plugin[l, k] = Gamma_plugin[k, l]

            # Bias-corrected estimators: correct.mat and Proj.array
            if self.verbose:
                print("======> Bias Correction for matrix Gamma....")

            correct_mat = np.zeros((self.L, self.L))
            Proj_array = np.zeros((self.L, self.L, self.d))

            for l in range(self.L):
                for k in range(self.L):
                    loading = (Sigma0 @ init_est[k]['beta_init']).reshape(1, -1)
                    Est_lk = bc.LF(self.X_list[l], self.y_list[l], loading,
                                beta_init=init_est[l]['beta_init'])
                    est_debias_vec = np.asarray(Est_lk['est_debias_vec']).ravel()
                    est_plugin_vec = np.asarray(Est_lk['est_plugin_vec']).ravel() if 'est_plugin_vec' in Est_lk else np.asarray(Est_lk.get('est.plugin.vec', est_debias_vec*0)).ravel()
                    correct_mat[l, k] = float(est_debias_vec[0] - est_plugin_vec[0])
                    Proj_array[l, k, :] = np.asarray(Est_lk['proj_mat']).ravel()
            self.Proj_array = Proj_array

            # Gamma.debias
            Gamma_debias = np.zeros((self.L, self.L))
            for l in range(self.L):
                for k in range(l, self.L):
                    Gamma_debias[l, k] = Gamma_plugin[l, k] + correct_mat[l, k] + correct_mat[k, l]
            for l in range(1, self.L):
                for k in range(0, l):
                    Gamma_debias[l, k] = Gamma_debias[k, l]

            self.Gamma_debias = Gamma_debias
            self.mean_Gamma_debias = self.Gamma_debias[np.tril_indices(self.L)]  ## self.mean_Gamma_debias the name need to change
            self.Gamma_plugin = Gamma_plugin


            ### Fitting DRO regression ###
            ## optimized weight vector
            self.weight_ = self.opt_weight(self.Gamma_debias)['weight']
            ## DRO regression coefficients
            self.est_bc = np.sum([self.debias_est[l]['est_debias_vec'] * self.weight_[l] for l in range(self.L)], axis=0)  # Shape: (n_loading,)

            self.beta_plug = np.sum([self.init_est[l]['beta_init'] * self.weight_[l] for l in range(self.L)], axis=0)
            self.est_plug = self.loading_mat @ self.beta_plug  # Shape: (n_loading,)
            self.parameters = {
                    'est_bc': self.est_bc,
                    'est_plug': self.est_plug,
                    'weight_': self.weight_
                }

    # ======================================================================= #
    # =================== Prediction  ======================================= #
    # ======================================================================= #
        def predict(self):
            """
            Predict using the fitted DRO regression model.

            Returns:
                pred (array-like): Predicted values for the target domain, shape (n0,).
            """

            pred = self.X0 @ self.beta_plug

            return pred

    # ======================================================================= #
    # =================== Compute CIs ======================================= #
    # ======================================================================= #
        def infer(self, M=200, alpha=0.05, alpha_thres=0.01):        
            """
            Perform resampling-based inference to compute confidence intervals for the loading coefficients.
            
            Args:
                M (int, optional): Number of resampling iterations. Defaults to 500.
                alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.
                alpha_thres (float, optional): Threshold for generating samples. Defaults to 0.01.  
            """
            if not hasattr(self, 'est_bc'):
                raise ValueError("Model is not fitted yet. Please call 'fit' first.")
            
            n_loading = self.loading_mat.shape[0]

            ### Sampling ###
            Var_Gamma = self.compute_Var_Gamma(tau=0.2)
            self.mean_Gamma_debias = self.mean_Gamma_debias.reshape(-1, 1)  # Ensure mu is a column vector
            gen_samples = self.gensamples(Var_Gamma, gen_size=M, threshold=0, alpha_thres=alpha_thres) # Shape: (M, gen_dim)

            gen_weight_mat = np.empty((M, self.L))

            for g in range(M):
                gen_matrix = np.full((self.L, self.L), np.nan)

                # Fill lower triangle and diagonal
                tril_indices = np.tril_indices(self.L)
                gen_matrix[tril_indices] = gen_samples[g, :]

                # Fill upper triangle by symmetry
                gen_matrix = gen_matrix + np.triu(gen_matrix.T, k=1)

                # Solve for optimal weights
                gen_sol = self.opt_weight(self.Gamma_debias)
                gen_weight_mat[g, :] = gen_sol["weight"]

            ### Constructing CIs ###
            CIs = np.zeros((n_loading, 2))
            for k in range(n_loading):
                loading_coef_0 = np.asarray([self.debias_est[l]['est_debias_vec'][k] for l in range(self.L)])
                gen_loading_coef_ = (gen_weight_mat @ loading_coef_0).reshape(-1)  # Shape: (M,)
                ses = np.asarray([self.debias_est[l]['se_vec'][k] for l in range(self.L)])  # Standard errors for each source domain, shape (L,)
                gen_se = (gen_weight_mat @ ses).reshape(-1)  # Shape: (M,)
                    # Compute confidence intervals
                z_alpha = norm.ppf(1 - alpha / 2)
                gen_CIs_lb = gen_loading_coef_ - z_alpha * gen_se # Shape: (M,)
                gen_CIs_ub = gen_loading_coef_ + z_alpha * gen_se # Shape: (M,)
                CIs[k, 0] = np.min(gen_CIs_lb)
                CIs[k, 1] = np.max(gen_CIs_ub)  
            self.CI = CIs


    # ======================================================================= #
    # =================== Summary Functions ================================= #
    # ======================================================================= #
        def summary(self):
            """
            Print a summary of the fitted model, including coefficients, weights, and CIs.

            Args:
                index (list or int, optional): Specific dimensions to display. Defaults to None (all dimensions).
            """
            if not hasattr(self, 'parameters'):
                raise ValueError("Model is not fitted yet. Please call 'fit' first.")

            # ---- helpers ----
            def _print_chunks(label, indices, values, width=8, per_row=10, fmt="{:>8.4f}", header_label="index"):
                """Pretty-print header+row in chunks."""
                # values is assumed aligned to indices order
                for start in range(0, len(indices), per_row):
                    chunk_idx  = indices[start:start+per_row]
                    chunk_vals = values[start:start+per_row]
                    header = f"{header_label:<10}| " + " ".join(f"{(i+1):>{width}}" for i in chunk_idx)
                    row    = f"{label:<10}| " + " ".join(fmt.format(v) for v in chunk_vals)
                    print(header)
                    print(row)

            # ---- data prep ----
            est_plug = self.parameters['est_plug']   # shape (n_loading,)
            est_bc   = self.parameters['est_bc']     # shape (n_loading,)
            d = len(est_bc)
            dim_idx = self.index if isinstance(self.index, list) else [self.index]

            print("Model Summary:")
            print("=================================")

            # ---- Weights (as table: group | 1..L) ----
            weight = self.parameters['weight_']  # shape (L,)
            L = len(weight)
            print("CGDRO Aggregated Weights:\n")
            group_idx = list(range(L))
            _print_chunks("weight_", group_idx, list(weight), width=8, per_row=10, fmt="{:>8.4f}", header_label="group")
            print()

            print("=================================")

            # ---- Plug-in estimates ----
            print("Plug-in Estimators:\n")
            plug_vals = est_plug
            _print_chunks("coef_", dim_idx, plug_vals, width=8, per_row=10, fmt="{:>8.4f}", header_label="index")
            print()

            print("=================================")
            print("Debiased Estimators:\n")
            bc_vals = est_bc
            _print_chunks("coef_", dim_idx, bc_vals, width=8, per_row=10, fmt="{:>8.4f}", header_label="index")
            print()

            # ---- Confidence Intervals ----
            if hasattr(self, 'CI'):
                CI = self.CI  # shape (n_loading, 2)
                print("=================================")
                print("Confidence Intervals:\n")

                # Pre-format as tuple strings; 5 per row for readability
                ci_strs = [f"({CI[i,0]:.4f},{CI[i,1]:.4f})" for i in range(len(dim_idx))]
                _print_chunks("CI", dim_idx, ci_strs, width=14, per_row=5, fmt="{}", header_label="index")
                print()
            else:
                print("Confidence Intervals not computed. Please run infer() method.")

    # ========================================================== #
    # =================== Internal Functions =================== #
    # ========================================================== #

        def opt_weight(self, Gamma, report_reward=False):
            """
            Compute Ridge-type weight vector using cvxpy.
            
            Args:
                Gamma (array-like): L x L matrix.
                report_reward (bool, optional): Whether to report the reward value. Defaults to False.
            Returns:
                dict: Contains 'weight' (optimal weights) and optionally 'reward' (optimal reward value).
            """
            Gamma = np.array(Gamma, dtype=float)
            L = Gamma.shape[1]
            
            # Eigen-decomposition and positive adjustment
            eigvals, eigvecs = np.linalg.eigh(Gamma)
            eigvals = np.maximum(eigvals, 0.001)  # ensure positive definite
            Gamma_positive = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Define variable
            v = cp.Variable(L, nonneg=True)
            objective = cp.Minimize(cp.quad_form(v, Gamma_positive + np.eye(L) * self.delta))
            constraints = [cp.sum(v) == 1]
            prob = cp.Problem(objective, constraints)
            
            prob.solve()
            opt_weight = v.value
            opt_weight[np.abs(opt_weight) <= 1e-8] = 0.0  # threshold
            

            return {"weight": opt_weight}


        def index_map(self, l, k):
            """
            Maps (l, k) with l >= k into vectorized index of upper-triangular part (column-wise),
            matching R: index.map(L, l, k) = (2L - k)(k - 1)/2 + l

            Args:
                l (int): row index (0-based)
                k (int): column index (0-based), must satisfy l >= k
            Returns:
                int: vectorized index (0-based)
            """
            return int((2 * self.L - (k + 1)) * k // 2 + l)

        def compute_Var_Gamma(self, tau=0.2):
            L = len(self.X_list)
            ns = np.array([X.shape[0] for X in self.X_list])  # Number of samples in each source domain
            gen_dim = L * (L + 1) // 2
            Var_Gamma = np.full((gen_dim, gen_dim), np.nan)

            for k1 in range(L):
                for l1 in range(k1, L):
                    for k2 in range(L):
                        for l2 in range(k2, L):
                            ind1 = self.index_map(l1, k1)
                            ind2 = self.index_map(l2, k2)

                            X_l1 = self.X_list[l1].copy()
                            X_k1 = self.X_list[k1].copy()

                            if self.intercept:
                                X_l1 = np.hstack((np.ones((X_l1.shape[0], 1)), X_l1))
                                X_k1 = np.hstack((np.ones((X_k1.shape[0], 1)), X_k1))

                            Sigma_l1 = X_l1.T @ X_l1 / X_l1.shape[0]
                            Sigma_k1 = X_k1.T @ X_k1 / X_k1.shape[0]

                            dev_l1 = self.init_est[l1]['dev']
                            dev_k1 = self.init_est[k1]['dev']
                        

                            # projection vectors
                            Proj1 = self.Proj_array[l1, k1,:]
                            Proj2_l1 = self.Proj_array[l2,k2,:] if l2 == l1 else np.zeros(self.d)
                            Proj2_k1 = self.Proj_array[k2,l2,:] if k2 == l1 else np.zeros(self.d)
                            val1 = dev_l1 / X_l1.shape[0] * (Proj1 @ Sigma_l1 @ (Proj2_l1 + Proj2_k1))
                        

                            Proj3 = self.Proj_array[k1, l1,:]
                            Proj4_l1 = self.Proj_array[l2, k2,:] if l2 == k1 else np.zeros(self.d)
                            Proj4_k1 = self.Proj_array[k2, l2,:] if k2 == k1 else np.zeros(self.d)
                            val2 = dev_k1 / X_k1.shape[0] * (Proj3 @ Sigma_k1 @ (Proj4_l1 + Proj4_k1))
                        

                            P1 = self.pred0_mat[:,k1] * self.pred0_mat[:,l1]
                            P2 = self.pred0_mat[:,k2] * self.pred0_mat[:,l2]
                            val3 = np.mean((P1 - np.mean(P1)) * (P2 - np.mean(P2))) / self.X0.shape[0]
                        
                        
                            val = val1 + val2 + val3
                            Var_Gamma[ind1, ind2] = val

            # Regularize the diagonal
            diag_correction = np.maximum(tau * np.diag(Var_Gamma), 1.0 / np.min(ns))
            Var_Gamma += np.diag(diag_correction)

            return Var_Gamma


        def gensamples(self, gen_Cov, gen_size=500, threshold=0, alpha_thres=0.01):
            """
            Generate samples from a multivariate normal distribution with different truncation strategies.

            Args:
                gen_Cov (array-like): Covariance matrix for the multivariate normal distribution.
                gen_size (int, optional): Number of samples to generate. Defaults to 500.   
                threshold (int, optional): Truncation strategy (0: coordinate-wise, 1: chi-square, 2: none). Defaults to 0.
                alpha_thres (float, optional): Significance level for truncation. Defaults to 0.01.
            Returns:
                array-like: Generated samples of shape (gen_size, gen_dim).
            """
            gen_mu = np.asarray(self.mean_Gamma_debias).reshape(-1)  # Ensure gen_mu is a 1D array
            gen_dim = len(gen_mu)
            gen_Cov = np.asarray(gen_Cov)

            # Output container
            gen_samples = np.zeros((gen_size, gen_dim))
            n_picked = 0

            # Threshold 0: coordinate-wise truncation based on Normal quantile
            if threshold == 0:
                thres = norm.ppf(1 - alpha_thres / (gen_dim * 2))
                while n_picked < gen_size:
                    S = np.random.multivariate_normal(mean=np.zeros(gen_dim), cov=gen_Cov)
                    if np.max(np.abs(S / np.sqrt(np.diag(gen_Cov)))) <= thres:
                        gen_samples[n_picked, :] = gen_mu + S
                        n_picked += 1

            # Threshold 1: chi-square truncation
            elif threshold == 1:
                U, D, Vt = svd(gen_Cov)
                D_sqrt = np.sqrt(D)
                gen_Cov_sqrt = U @ np.diag(D_sqrt) @ Vt
                thres = chi2.ppf(1 - alpha_thres, df=gen_dim)
                while n_picked < gen_size:
                    Z = np.random.normal(size=gen_dim)
                    Z_normsq = np.sum(Z**2)
                    if Z_normsq <= thres:
                        gen_samples[n_picked, :] = gen_mu + gen_Cov_sqrt @ Z
                        n_picked += 1

            # Threshold 2: no threshold (standard MVN sampling)
            elif threshold == 2:
                gen_samples = np.random.multivariate_normal(mean=gen_mu, cov=gen_Cov, size=gen_size)

            else:
                raise ValueError("threshold must be 0, 1, or 2.")

            return gen_samples


        


        def dev_fun(self, pred, y, sparsity=0):
            pred = np.asarray(pred).ravel()  
            y = np.asarray(y).ravel()
            n = len(y)
            sigmasq_hat = np.sum((y - pred) ** 2) / max(0.7 * n, n - sparsity)
            return sigmasq_hat
















class ml:
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

    def fit(self, X_list, y_list, X0=None, loss_type='reward', bias_correct=True, priors=None):
        """Compute the plug-in and bias-corrected estimators of the Gamma matrix.

        Args:
            X_list (list): list of feature matrices on each source domain
            y_list (list): list of label arrays on each source domain
            X0 (array, optional): feature matrix on the target domain. If None, use
                                    the pooled source data as the target data. Defaults to None.
            loss_type (str, optional): type of the loss function used to compute the optimal aggregation weights.
                                    Options include 'reward' (default), 'squaredloss', and 'regret'. Defaults to 'reward'.
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
        dev_vec = np.zeros(self.L)
        self.source_full_models = [UtilModels(mode='reg', f_learner=self.f_learner, w_learner=self.w_learner, split=False, seed=self.seed, verbose=self.verbose) for l in range(self.L)]
        for l in range(self.L):
            self.source_full_models[l].fit_f(self.X_list[l], self.y_list[l])
            self.pred_full_mat[:, l] = self.source_full_models[l].model_f.predict(self.X0)
            pred_l = self.source_full_models[l].model_f.predict(self.X_list[l])
            dev_vec[l] = np.sum((self.y_list[l] - pred_l) ** 2) / (nl_s[l] - self.d)
        # The plug-in estimator of Gamma matrix
        self.Gamma_plug = self.pred_full_mat.T @ self.pred_full_mat / N
        self.dev_vec = dev_vec

        
       

        



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
        
        weight_sol = self.opt_weight(Gamma, loss_type=loss_type, priors=priors)
        self.weight_ = weight_sol['weight']
        
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
    
    def opt_weight(self, Gamma, loss_type='reward', priors=None):
            """
            Compute Ridge-type weight vector using cvxpy.
            
            Args:
                Gamma (array-like): L x L matrix.
                loss_type (str, optional): Type of loss function ('reward', 'squaredloss or 'regret'). Defaults to 'reward'.
                priors (tuple, optional): prior information on the aggregation weights, given as (prior_weight, rho),
                                    where prior_weight is the prior weight vector and rho is the radius of the
                                    L2-norm ball around prior_weight. If None, no prior information is used. Defaults to None.
            Returns:
                dict: Contains 'weight' (optimal weights) and optionally 'reward' (optimal reward value).
            """
            Gamma = np.array(Gamma, dtype=float)
            Gamma_diag = cp.diag(Gamma)
            L = Gamma.shape[1]
            
            # Eigen-decomposition and positive adjustment
            eigvals, eigvecs = np.linalg.eigh(Gamma)
            eigvals = np.maximum(eigvals, 0.001)  # ensure positive definite
            Gamma_positive = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Define variable
            v = cp.Variable(L, nonneg=True)
            if loss_type == 'reward':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive))
            elif loss_type == 'squaredloss':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive) - cp.matmul(v, Gamma_diag + self.dev_vec))
            elif loss_type == 'regret':
                objective = cp.Minimize(cp.quad_form(v, Gamma_positive) - cp.matmul(v, Gamma_diag))

            if priors is None:
                constraints = [cp.sum(v) == 1]
            else:
                prior_weight, rho = priors
                constraints = [cp.sum(v) == 1, cp.norm(v - prior_weight) <= rho]
            constraints = [cp.sum(v) == 1]
            prob = cp.Problem(objective, constraints)
            

            try:
                prob.solve()
                opt_weight = v.value
                opt_weight[np.abs(opt_weight) <= 1e-8] = 0.0  # threshold
            except (DCPError, SolverError) as e:
                print("f_learner or w_learner is not well learned.")
                opt_weight = None

            return {"weight": opt_weight}


    

        