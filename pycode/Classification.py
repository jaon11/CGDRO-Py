import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.linalg import svd
import cvxpy as cp
from utils import UtilModels, compute_softmax, fit_fw, BiasCorrection
from concurrent.futures import ThreadPoolExecutor, as_completed



class linear: 
    """
    CGDRO: Classification for linear model (Cross-Entropy Loss)

    Args:
        f_learner (str, optional): method used to fit outcome models on each source. Defaults to 'linear'.
        w_learner (str, optional): method used to fit density models on each source. Defaults to 'linear'.
        split (bool, optional): whether to split the source data into two halves for fitting outcome and density models. Defaults to True.
        seed (int, optional): random seed. Defaults to 123.
    """

    def __init__(self, f_learner='linear', w_learner='linear',split=True, seed=123):
        self.f_learner = f_learner
        self.w_learner = w_learner
        self.split = split
        self.seed = seed
        self.log_message = []


# ==================================================================================================== #
# =================== Run Optimistic Gradient Mirror Prox to solve gamma and theta =================== #
# ==================================================================================================== #
    def fit(self, X_list, y_list, X0=None, max_iter=1000, tol=1e-6, check_dual=False, verbose=False):
        """
        Estimate the parameters theta and gamma using Optimistic Gradient Mirror Prox.
        
        Args:
            X_list (list): list of feature matrices on each source domain
            y_list (list): list of label arrays on each source domain
            X0 (array, optional): feature matrix on the target domain. If None, use
                                the pooled source data as the target data. Defaults to None.    
            max_iter (int, optional): maximum number of iterations. Defaults to 1000.
            tol (float, optional): tolerance for convergence. Defaults to 1e-6.
            check_dual (bool, optional): whether to check the duality gap. Defaults to False.
            verbose (bool, optional): whether to print out the fitting information. Defaults to False.
        """

        self.X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]  # List of source domain features
        self.y_list = [np.asarray(yi, dtype=int).ravel() for yi in y_list]  # List of source domain labels
        if X0 is None:
            X0 = np.vstack(self.X_list)  # Target domain features
        else:
            X0 = X0          # Target domain features
        X0 = np.asarray(X0, dtype=float)            
        X0 = X0
        self.X0 = X0


        self.L = len(X_list)  # Number of source domains
        self.d = X_list[0].shape[1]    # Feature dimension
        self.num_class = len(np.unique(y_list[0]))  # Number of classes

        # Initialize the models with (tuned) hyperparameters
        self.models = UtilModels("cls", self.f_learner, self.w_learner, self.split, self.seed, verbose)

        self.fit_mu() ## get the value of mu_list with function fit()
        self.theta = np.zeros(self.d * (self.num_class - 1)) 
        self.gamma = np.ones(self.L) / self.L 
        theta_bar = self.theta.copy()
        gamma_bar = self.gamma.copy()

        # parameters for adaptive learning rate
        eta = np.sqrt(2)
        a = 1.2
        b = np.log(self.L)
        Z_cumsum = 0. 
        
        # parameters for logging
        primal = self._compute_primal(self.theta)
        
        # optimization loop
        for iter in range(max_iter):
            # --------- Intermediate Step --------- #
            grad_theta_bar, grad_gamma_bar = self._compute_grad(theta_bar, gamma_bar)
            theta_bar = self.theta - (eta / a) * grad_theta_bar
            gamma_bar = self.gamma * np.exp(eta / b * grad_gamma_bar)
            gamma_bar /= gamma_bar.sum()
            
            # --------- Correction Step --------- #
            grad_theta_bar, grad_gamma_bar = self._compute_grad(theta_bar, gamma_bar)
            theta_curr = theta_bar - (eta / a) * grad_theta_bar
            gamma_curr = gamma_bar * np.exp(eta / b * grad_gamma_bar)
            gamma_curr /= gamma_curr.sum()
            
            # --------- Adaptive Learning Rate --------- #
            Z = a * (np.linalg.norm(theta_bar - theta_curr) ** 2 
                + np.linalg.norm(theta_bar - self.theta) ** 2) + \
                    b * (np.linalg.norm(gamma_bar - gamma_curr, ord=1) ** 2
                        + np.linalg.norm(gamma_bar - self.gamma, ord=1) ** 2)
            Z_cumsum += Z / (5 * eta ** 2)
            eta = 1 * np.sqrt(2) / np.sqrt(1 + Z_cumsum)
            
            # --------- Update Parameters --------- #
            self.theta = theta_curr
            self.gamma = gamma_curr
            
            primal_curr = self._compute_primal(self.theta)
            # Check duality
            if iter % 50 == 0:
                if check_dual:
                    dual = self._compute_dual(self.gamma)
                    dual_gap = np.abs(primal - dual)
                    log_info = f"Iter {iter+1} | Diff primal: {np.abs(primal - primal_curr):.6f} | Dual gap: {dual_gap:.6f}"
                else:
                    log_info = f"Iter {iter+1} | Diff primal: {np.abs(primal - primal_curr):.6f}"
                self.log_message.append(log_info)
                if verbose:
                    print(log_info)    
            
            # Check convergence
            if np.abs(primal_curr - primal) < tol:
                if verbose:
                    print(f"Converged at iteration {iter+1} with Primal gap {np.abs(primal - primal_curr):.6f}.")
                break

            primal = primal_curr
            theta_mat = self.theta.reshape(-1, self.d).T  # Reshape theta to a matrix of shape (d, num_class-1)
            self.parameters = {
                'coef_': theta_mat,
                'weight_': self.gamma
            }


# ======================================================================= #
# =================== Prediction  ======================================= #
# ======================================================================= #
    def predict_proba(self):
        """
        Predict the probabilities of each class for the given input X.
        """ 

        theta_mat = self.theta.reshape(-1, self.d).T
        theta_mat = np.column_stack([np.zeros(self.d), theta_mat])  # Add zero column for the reference class
        logits = self.X0 @ theta_mat  # Shape (n_samples, num_class)
        logits_max = np.max(logits, axis=1, keepdims=True)
        stable_logits = logits - logits_max  # subtract max for numerical stability
        exp_terms = np.exp(stable_logits)
        proba = exp_terms / (exp_terms.sum(axis=1, keepdims=True))  # Shape (n_samples, num_class)
        proba = np.hstack([proba])  # Add the reference class
        return proba

    def predict(self):
        """
        Predict the class labels for the given input X.
        """

        proba = self.predict_proba()
        pred = np.argmax(proba, axis=1)
        return pred

# ======================================================================= #
# =================== Compute CIs ======================================= #
# ======================================================================= #
    def infer(self, M=200, alpha=0.05, diag=True, parallel=False, n_workers=4):
        """
        Performs resampling for inference.

        Args:
            M (int, optional): number of resampling iterations. Defaults to 500.
            alpha (float, optional): significance level for confidence intervals. Defaults to 0.05.
            diag (bool, optional): whether to use diagonal approximation for covariance matrices. Defaults to True.
            parallel (bool, optional): whether to use parallel computing. Defaults to False.
            n_workers (int, optional): number of workers for parallel computing. Defaults to 4.
        """
        ## Inference
        self.gradS = None           # Gradient of S
        self.H_inv = None           # Hessian inverse of S
        # Covariance matrices
        self.mu_cov_list = []       # Covariance matrices of mu
        self.gradS_cov = None       # Covariance matrix of gradS
        self.mu_gradS_cov_list = [] # Covariance matrices of mu and gradS    
        self.theta_M = []           # Resampled theta
        self.gamma_M = []           # Resampled gamma     
        self.CI_lb_M = []           # Resampled Lower bound of CI for theta
        self.CI_ub_M = []           # Resampled Upper bound of CI for theta   

        # Prepare materials
        self._prepare(diag=diag)
        
        def resample_and_compute(diag=diag):
            # Generate resampled mu values
            mu_resample_list = [np.random.multivariate_normal(mu, cov)
                                for mu, cov in zip(self.mu_list, self.mu_cov_list)]
            # Solve the optimization problem
            theta_resample, gamma_resample = self._solve_resample(mu_resample_list, diag=diag)
            # Compute variance of theta_resample
            var_theta_resample = self._compute_variance_resample(gamma_resample)
            
            # Calculate single 95% ci
            z_alpha = norm.ppf(1 - alpha / 2)
            CI_lb = theta_resample - z_alpha * np.sqrt(np.diag(var_theta_resample))
            CI_ub = theta_resample + z_alpha * np.sqrt(np.diag(var_theta_resample))

            return theta_resample, gamma_resample, CI_lb, CI_ub
        
        if parallel:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(resample_and_compute) for _ in range(M)]
                for future in as_completed(futures):
                    theta_resample, gamma_resample, CI_lb, CI_ub = future.result()
                    self.theta_M.append(theta_resample)
                    self.gamma_M.append(gamma_resample)
                    self.CI_lb_M.append(CI_lb)
                    self.CI_ub_M.append(CI_ub)
        else:
            for _ in range(M):
                theta_resample, gamma_resample, CI_lb, CI_ub = resample_and_compute()
                self.theta_M.append(theta_resample)
                self.gamma_M.append(gamma_resample)
                self.CI_lb_M.append(CI_lb)
                self.CI_ub_M.append(CI_ub)
        
        self.CI_lb_U = np.min(self.CI_lb_M,axis=0)
        self.CI_ub_U = np.max(self.CI_ub_M,axis=0)
        K = self.num_class - 1
        CI_Union = [(float(round(self.CI_lb_U[i], 4)), float(round(self.CI_ub_U[i], 4))) for i in range(self.d*K)]
        CI_Union = list(zip(*[iter(CI_Union)] * self.d))
        self.CI = np.swapaxes(np.array(CI_Union), 0, 1) # (d,K,2)
            



# ========================================================== #
# =================== Summary Functions =================== #
# ========================================================== #
    def summary(self, index=None, class_index=None):
        """
        Print the summary of the fitted model.

        Args:
            index (array-like or None): 1-based indices of dimensions to print (subset of 1..d).
                                            Defaults to all dimensions.
            class_index (array-like or None): class labels to print (subset of 2..self.num_class).
                                            Defaults to all (2..self.num_class).
        """

        if not hasattr(self, 'parameters'):
            print("Model is not fitted yet. Please call the 'fit' method first.")
            return

        # ---- helpers ----
        def _normalize_indices(user_idx, lo_1based, hi_1based, name):
            if user_idx is None:
                return list(range(lo_1based - 1, hi_1based))  # all valid
            try:
                idx_list = list(user_idx)
            except TypeError:
                idx_list = [user_idx]
            norm = []
            for v in idx_list:
                vv = int(v)
                if not (lo_1based <= vv <= hi_1based):
                    raise ValueError(f"{name} out of range: {vv} not in [{lo_1based}, {hi_1based}]")
                norm.append(vv - 1)
            # dedup preserving order
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
        # ---- summary header ----
        print("Model Summary:")
        print("=================================")

        # ---- weights ----
        weight = self.parameters['weight_']  # shape (L,)
        L = len(weight)
        group_idx = list(range(L))
        print("Fitted Weights:\n")
        _print_chunks("weight_", group_idx, list(weight), width=8, per_row=10, fmt="{:>8.4f}", header_label="group")
        print()

        print("=================================")
        print("Fitted Coefficients:\n")

        # ---- coefficients ----
        coef = self.parameters['coef_']  # shape (d, K)
        d, K = coef.shape
        dim_idx = _normalize_indices(index, 1, d, "index")

        if class_index is None:
            class_list = list(range(1, self.num_class))
        else:
            try:
                class_list = list(class_index)
            except TypeError:
                class_list = [class_index]

        for c in class_list:
            if not (1 <= c <= self.num_class - 1):
                raise ValueError(f"class_index out of range: {c} not in [1, {self.num_class-1}]")
            j = c - 1
            coef_j = coef[dim_idx, j].ravel()

            print(f"Class {c} coefficients:")
            _print_chunks("coef_", dim_idx, list(coef_j), width=8, per_row=10, fmt="{:>8.4f}")
            print()

        # ---- confidence intervals ----
        if hasattr(self, 'CI'):
            print("=================================")
            print("Confidence Intervals for each coefficient:")
            CI = self.CI  # shape (d, K, 2)

            for c in class_list:
                j = c - 1
                ci_j = CI[dim_idx, j, :]  # shape (len(dim_idx), 2)
                print(f"\nClass {c} Confidence Intervals:")

                # pre-format CIs as strings
                ci_strs = [f"({low:.3f},{high:.3f})" for (low, high) in ci_j]
                _print_chunks("CIs", dim_idx, ci_strs, width=14, per_row=5, fmt="{}")
            print()
        else:
            print("Confidence Intervals not computed. Please run infer() method.")

# ========================================================== #    
# =================== Internal Functions =================== #
# ========================================================== #
    def fit_mu(self):
        """Compute the doubly-robust estimator of mu.
        Returns:
            mu_list: a list of length L. Each element is a vector of length num_class.
        """
        self.probaX_list, self.probaX0_list, self.omegaX_list = fit_fw(self.models, self.X_list, self.y_list, self.X0)

        mu_list = []
        for l in range(self.L):
            n_l = self.X_list[l].shape[0]
            n0 = self.X0.shape[0]
            
            # Create one-hot representation for y (for all classes)
            y_onehot = np.eye(self.num_class)[self.y_list[l]]
            
            # ------------ Compute mu ----------- #
            
            term1 = - self.X0.T @ self.probaX0_list[l][:, 1:] / n0  # (d, N) * (N, num_class-1) = (d, num_class-1)
            term1_flat = term1.flatten(order='F') # shape: (d*(num_class-1),)
            # Notice that each term1's column corresponds to empirical average of -f_c(X)X, where c denotes the class.
            # Therefore, we need to flatten it by column-wise.
            
            # (d, n_l) * (n_l, num_class-1) = (d, num_class-1)
            term2 = - (self.X_list[l] * self.omegaX_list[l][:, np.newaxis]).T @ (y_onehot[:, 1:] - self.probaX_list[l][:, 1:]) / n_l
            term2_flat = term2.flatten(order='F') # shape: (d*(num_class-1),)
            # Notice that each term2's column corresponds to empirical average of -w(X)(1_c - f_c(X))X, where c denotes the class.
            # Therefore, we need to flatten it by column-wise.
            mu = term1_flat + term2_flat
            mu_list.append(mu)

        self.mu_list = mu_list

    def _compute_primal(self, theta):
        """Compute the value of the primal problem.
        Primal Problem: max_{gamma} obj(theta, gamma)
        This function is used for evaluation to tell if the algorithm converges.
        Args:
            theta: a vector of length d * (num_class - 1).
        """
        g_theta = np.max([np.dot(theta, mu) for mu in self.mu_list])
        theta_mat = theta.reshape(-1, self.d).T # Reshape theta to a matrix of shape (d, num_class-1)
        # Compute the stable logitis (logits - max(logits)) for numerical stability
        logits = self.X0 @ theta_mat
        logits_max = np.max(logits, axis=1, keepdims=True)
        stable_logits = logits - logits_max  # subtract max for numerical stability
        exp_terms = np.exp(stable_logits)
        S_theta = np.mean(logits_max + np.log(np.exp(-logits_max) + np.sum(exp_terms, axis=1)))
        
        return g_theta + S_theta
    
    def _compute_dual(self, gamma):
        """Compute the value of the dual problem (after swapping minimax).
        Dual Problem: min_{theta} obj(theta, gamma)
        This function is used for internal checking if the algorithm converges.
        Arg:
            gamma: a vector of length num_class-1. Corresponding to the weights of each source domain.
        """
        
        def f(theta):
            obj = np.sum(np.array([gamma[l] * np.dot(theta, self.mu_list[l]) for l in range(self.L)]))
            theta_mat = theta.reshape(-1, self.d).T # Reshape theta to a matrix of shape (d, num_class-1)
            # Compute the stable logitis (logits - max(logits)) for numerical stability
            logits = self.X0 @ theta_mat
            logits_max = np.max(logits, axis=1, keepdims=True)
            stable_logits = logits - logits_max  # subtract max for numerical stability
            exp_terms = np.exp(stable_logits)
            obj += np.mean(logits_max + np.log(np.exp(-logits_max) + np.sum(exp_terms, axis=1)))
            # obj += np.mean(np.log(1 + np.sum(np.exp(self.X0 @ theta_mat), axis=1)))
            return obj
    
        result_min_theta = minimize(f, self.theta, method='L-BFGS-B')
        return result_min_theta.fun
    
    def _compute_grad(self, theta, gamma):
        """Compute the gradient of the objective function w.r.t. theta.
        theta: a vector of length d * (num_class - 1).
        gamma: a vector of length num_class-1.
        """
        # ---------- Compute the gradient in terms of theta --------- #
        # Compute gradient of S(\theta)
        theta_mat = theta.reshape(-1, self.d).T # Reshape theta to a matrix of shape (d, num_class-1)
        proba_mat = compute_softmax(self.X0 @ theta_mat) # (N, num_class - 1)
        grad_S = (self.X0.T @ proba_mat / self.X0.shape[0]).flatten(order='F') # vector of length d * (num_class - 1)
        # Combine the gradient for obj in terms of theta
        grad_theta = grad_S + np.sum([gamma[l] * self.mu_list[l] for l in range(self.L)], axis=0)
        
        # --------- Compute the gradient in terms of gamma --------- #
        grad_gamma = np.array([np.dot(theta, mu) for mu in self.mu_list])
        
        return grad_theta, grad_gamma
        
    def _prepare(self, diag=True):  # parameters: probaX_list, probaX0_list, omegaX_list can be computed first?
        """Prepares materials for inference.
        Compute and store the Hessian inverse, gradient of S, and covariance matrices.
        """
        n0 = self.X0.shape[0]
        try:
            theta_mat = self.theta.reshape(-1, self.d).T
        except AttributeError as e:
            print("Parameters need to estimate first.")
        proba_mat = compute_softmax(self.X0 @ theta_mat) # (N, num_class - 1)
        
        # ------- Hessian Matrix -------
        d = self.d
        K = self.num_class - 1
        H = np.zeros((d * K, d * K))
        for j in range(K):
            for k in range(K):
                pj, pk = proba_mat[:, j], proba_mat[:, k]
                weights = pj * ((j==k) - pk)
                H_block = self.X0.T @ np.diag(weights) @ self.X0 / n0
                H[j*d:(j+1)*d, k*d:(k+1)*d] = H_block
        self.H_inv = np.linalg.inv(H) if not diag else np.diag(1/np.diag(H))
        
        # -------- Covariance Calculations --------
        # Compute gradS's covariance
        self.gradS = (self.X0.T @ proba_mat / n0).flatten(order='F') # vector of length d * (num_class - 1)
        def diag_cov(psi):
            return np.diag(np.mean(psi**2, axis=0) - np.mean(psi, axis=0)**2)
        psiS = np.array([np.kron(proba_mat[i,:], self.X0[i]) for i in range(n0)])
        self.gradS_cov = (diag_cov(psiS) if diag else np.cov(psiS, rowvar=False)) / n0
        
        # Compute mu's covariance and its covariance with gradS for each source l
        for l in range(self.L):
            n_l = self.X_list[l].shape[0]
            n0 = self.X0.shape[0]
            y_onehot = np.eye(self.num_class)[self.y_list[l]]
            
            # Term 1: X0 contribution
            psi1 = np.array([-np.kron(self.probaX0_list[l][i, 1:], self.X0[i])
                            for i in range(n0)])
            cov_psi1 = (diag_cov(psi1) if diag else np.cov(psi1, rowvar=False)) / n0
            
            # Term 2: Xl contribution
            psi2 = np.array([-np.kron(y_onehot[i, 1:] - self.probaX_list[l][i, 1:],
                                    self.X_list[l][i] * self.omegaX_list[l][i])
                            for i in range(n_l)])
            cov_psi2 = (diag_cov(psi2) if diag else np.cov(psi2, rowvar=False)) / n_l
        
            self.mu_cov_list.append(cov_psi1 + cov_psi2)
            
            # cross-covariance
            if diag:
                cov_diag = np.mean(psi1 * psiS, axis=0) - np.mean(psi1, axis=0) * np.mean(psiS, axis=0)
                mu_gradS_cov = np.diag(cov_diag)/n0
            else:
                mu_gradS_cov = np.cov(psi1, psiS, rowvar=False)[:d*K, d*K:]/n0
            self.mu_gradS_cov_list.append(mu_gradS_cov)
    
    def _solve_resample(self, mu_resample_list, diag):
        """Solve the resampled gamma and resampled theta for a resampled mu."""
        H_inv = self.H_inv
        if diag and H_inv.ndim == 2:
            H_inv = np.diag(H_inv)
        
        def obj_gamma(gamma):
            weighted_mu = np.average(mu_resample_list, axis=0, weights=gamma)
            g = weighted_mu + self.gradS
            if diag:
                quad_term = 0.5 * np.sum(g ** 2 * H_inv)
            else:
                quad_term = 0.5 * g @ H_inv @ g
            linear_term = - np.dot(gamma, [mu @ self.theta for mu in mu_resample_list])
            return quad_term - linear_term
            # g = np.sum([gamma[l] * mu_resample_list[l] 
            #             for l in range(self.L)], axis=0) + self.gradS
            # return np.dot(g, self.H_inv @ g) * 0.5 \
            #         - np.sum([gamma[l] * np.dot(mu_resample_list[l], self.theta)
            #                     for l in range(self.L)])
        def grad_gamma(gamma):
            weighted_mu = np.average(mu_resample_list, axis=0, weights=gamma)
            g = weighted_mu + self.gradS
            if diag:
                H_g = g * H_inv
            else:
                H_g = H_inv @ g
            return np.array([
                np.dot(H_g, mu) - np.dot(mu, self.theta)
                for mu in mu_resample_list
            ])

        # Solve with analytical gradients
        bounds = [(0, 1) for _ in range(self.L)]    
        cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        result = minimize(obj_gamma, self.gamma, method='SLSQP', 
                        jac=grad_gamma, bounds=bounds, constraints=cons)
        gamma_resample = result.x
        
        self.weighted_mu_sum = np.sum([
            gamma_resample[l] * mu_resample_list[l] 
            for l in range(self.L)
        ], axis=0)
        
        def obj_theta(theta):
            theta_mat = theta.reshape(-1, self.d).T
            # Precompute matrix product using optimized BLAS operations
            X_theta = self.X0 @ theta_mat  # Shape (n_samples, num_class-1)
            
            # Numerically stable log-sum-exp with reference class
            max_vals = np.maximum(0, X_theta.max(axis=1))  # Shape (n_samples,)
            safe_exp = np.exp(X_theta - max_vals[:, None])
            log_sum = max_vals + np.log(1 + safe_exp.sum(axis=1))  # log(1 + sum(exp(η)))
            log_term = np.mean(log_sum)
            # Vectorized linear term calculation
            linear_term = np.dot(theta, self.weighted_mu_sum)
            return linear_term + log_term
        
        def grad_theta(theta):
            theta_mat = theta.reshape(-1, self.d).T
            X = self.X0  # Shape (n_samples, d)
            
            # Compute probabilities (softmax derivatives)
            X_theta = X @ theta_mat  # Shape (n_samples, num_class-1)
            max_vals = np.maximum(0, X_theta.max(axis=1, keepdims=True))
            exp_theta = np.exp(X_theta - max_vals)
            probs = exp_theta / (1 + exp_theta.sum(axis=1, keepdims=True))  # Shape (n_samples, num_class-1)
            # Gradient of log term: X^T @ probs / n_samples
            log_grad = (X.T @ probs).flatten(order='F') / X.shape[0]
            # Gradient of linear term: weighted_mu_sum
            linear_grad = self.weighted_mu_sum
            return linear_grad + log_grad
        
        result = minimize(obj_theta, self.theta, method='L-BFGS-B', jac=grad_theta)
        theta_resample = result.x
        
        return theta_resample, gamma_resample
    
    def _compute_variance_resample(self, gamma_resample):
        """Compute the variance of theta_resample."""
        term1 = np.sum([gamma_resample[l]**2 * self.mu_cov_list[l] for l in range(self.L)], axis=0)
        term2 = self.gradS_cov
        term3 = np.sum([gamma_resample[l] * self.mu_gradS_cov_list[l] for l in range(self.L)], axis=0)
        W_resample = term1 + term2 - 2 * term3 # + 1e-8 * np.eye(self.d * (self.num_class - 1))
        return self.H_inv @ W_resample @ self.H_inv
