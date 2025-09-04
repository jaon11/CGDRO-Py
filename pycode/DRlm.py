import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.linalg import svd
import cvxpy as cp
from utils import UtilModels, compute_softmax, fit_fw, BiasCorrection
from concurrent.futures import ThreadPoolExecutor, as_completed


class DRlm:
    class Classification: ## some global attributes in infer() would be changed to functional.
        '''
        CGDRO: Classification for linear model (Cross-Entropy Loss)

        Attributes:
            f_learner: the probability learner model.
            w_learner: the density ratio learner model.
            split: whether to split the source data into halves.
            models: a list of UtilModels for each source domain.
            mu_list: a list of length L. Each element is a vector of length num_class.
            theta: the final theta parameters.
            gamma: the final gamma parameters.
            theta_M: Resampled theta in inference procedure.
            gamma_M: Resampled gamma in inference procedure.
            CI_lb_M: the lower bounds of resampled CIs.
            CI_ub_M: the upper bounds of resampled CIs.
            CI_lb_U: the lower bound of the union CI.
            CI_ub_U: the upper bound of the union CI.
            CI_U: the union CI.
        '''
        def __init__(self, f_learner='linear', w_learner='linear',split=True, seed=123):
            self.f_learner = f_learner
            self.w_learner = w_learner
            self.split = split
            self.seed = seed
            self.log_message = []

            ## Inference
            #self.gradS = None           # Gradient of S
            #self.H_inv = None           # Hessian inverse of S
            # Covariance matrices
            #self.mu_cov_list = []       # Covariance matrices of mu
            #self.gradS_cov = None       # Covariance matrix of gradS
            #self.mu_gradS_cov_list = [] # Covariance matrices of mu and gradS
            # Resampling results
            #self.theta_M = []           # Resampled theta
            #self.gamma_M = []           # Resampled gamma
            #self.CI_lb_M = []           # Resampled Lower bound of CI for theta
            #self.CI_ub_M = []           # Resampled Upper bound of CI for theta
            #self.CI_lb_U = None         # Union Lower bound of CI for theta
            #self.CI_ub_U = None         # Union Upper bound of CI for theta
            #self.CI_U = None            # Union CI for theta

    # ==================================================================================================== #
    # =================== Run Optimistic Gradient Mirror Prox to solve gamma and theta =================== #
    # ==================================================================================================== #
        def fit(self, X_list, y_list, X0=None, max_iter=1000, tol=1e-6, check_dual=False, verbose=False):
            '''
            Estimate the parameters theta and gamma using Optimistic Gradient Mirror Prox.
            
            Attributes:
                X_list: a list of length L. Each element is a matrix of shape (n_l, d).
                y_list: a list of length L. Each element is a vector of length n_l.
                X0: a matrix of shape (n0, d). If None, it will be set to the vertical stack of X_list.
                max_iter: maximum number of iterations.
                tol: tolerance for convergence.
                check_dual: whether to check the duality gap during optimization.
                verbose: whether to print the log messages.
            '''
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
            '''
            Predict the probabilities of each class for the given input X.
            '''

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
            '''
            Predict the class labels for the given input X.
            '''

            proba = self.predict_proba()
            return np.argmax(proba, axis=1)

    # ======================================================================= #
    # =================== Compute CIs ======================================= #
    # ======================================================================= #
        def infer(self, index=0, M=500, alpha=0.05, diag=True, parallel=False, n_workers=4):
            '''
            Performs resampling for inference (coordinate = index).

            Attributes:
            index: the coordinate index for which to compute the confidence interval (dimension).
            M: Resampling times.
            parallel: parallel computing via CPU.
            n_workers: the number of workers in parallel computing via CPU.
            diag: whether to keep only the diagonal elements of the covariance matrices. 
                    (Recommended for element-wise inference)
            '''
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
            self.CI_U = np.swapaxes(np.array(CI_Union), 0, 1) # (d,K,2)
            self.CI_index = self.CI_U[index]  # CI for the coordinate index
                







    # ========================================================== #    
    # =================== Internal Functions =================== #
    # ========================================================== #
        def fit_mu(self, verbose=False):
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

    class Regression:
        '''
        Closed-form DRO linear regression (low-dimension case so far)
        '''
        def __init__(self,intercept=False,loading_intercept=False, delta=0, lam=None, verbose=False):
            self.intercept = intercept
            self.loading_intercept = loading_intercept
            self.lam = lam
            self.verbose = verbose
            self.delta = delta

    # ==================================================================================================== #
    # =================== Run Closed-form solution to solve gamma and theta =================== #
    # ==================================================================================================== #
        def fit(self, X_list, y_list, loading_mat, X0=None):
            """
            Fit (point estimate) the linear regression model using closed-form DRO.

            Attributes:
            X_list : list of np.ndarray
                List of source domain features, each of shape (n_l, d).
            y_list : list of np.ndarray
                List of source domain labels, each of shape (n_l,).
            X0 : np.ndarray, optional
                Target domain features, shape (n0, d). If None, uses all source data
            loading_mat : np.ndarray
                Loading matrix for coefficients, shape (n_loading, d). 
            delta : float, optional
                Ridge penalty level, non-positive (default is 0).
            Returns
            -------
            self : object
                Fitted estimator.   

            """
            self.X_list = [np.asarray(Xi, dtype=float) for Xi in X_list]
            self.y_list = [np.asarray(yi, dtype=float).ravel() for yi in y_list]
            self.loading_mat = np.asarray(loading_mat, dtype=float)
            self.L = len(self.X_list)  # Number of source domains
            self.d = self.X_list[0].shape[1]  + (1 if self.intercept else 0)  # Feature dimension
            bc = BiasCorrection(
                lam=self.lam,                         # if you have these attrs; otherwise omit
                intercept=self.intercept,
                loading_intercept=self.loading_intercept,
                verbose=self.verbose
            )



            if not isinstance(self.verbose, bool):
                self.verbose = True
            if (not self.intercept) and self.loading_intercept:
                self.loading_intercept = False
            if self.verbose:
                print("Argument 'loading_intercept' set to False because intercept is False")




            print('start fitting-----')
            ### Fitting Bias-corrected Estimator of Coef_ with loading matrix ###
            if self.verbose:
                print("======> Bias Correction for initial estimators....")

            fits_info = [None] * self.L
            Points_info = [None] * self.L

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

                fits_info[l] = {'beta_init': beta_init, 'dev': dev}
                Points_info[l] = {'est_debias_vec': np.asarray(Est['est_debias_vec']),
                                'se_vec': np.asarray(Est['se_vec'])}  
            self.fits_info = fits_info
            self.Points_info = Points_info

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
                pred0_mat[:, l] = (X0 @ self.fits_info[l]['beta_init']).ravel()
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
                    b_l = fits_info[l]['beta_init']
                    b_k = fits_info[k]['beta_init']
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
                    loading = (Sigma0 @ fits_info[k]['beta_init']).reshape(1, -1)
                    Est_lk = bc.LF(self.X_list[l], self.y_list[l], loading,
                                beta_init=fits_info[l]['beta_init'])
                    # Est_lk expects loading_mat as matrix; we pass single-column loading
                    # Est_lk returns est.debias.vec and est.plugin.vec arrays (length n_loading)
                    # Here original R code used Est.lk$est.debias.vec - Est.lk$est.plugin.vec
                    # When loading is single column, these vectors have length 1.
                    est_debias_vec = np.asarray(Est_lk['est_debias_vec']).ravel()
                    est_plugin_vec = np.asarray(Est_lk['est_plugin_vec']).ravel() if 'est_plugin_vec' in Est_lk else np.asarray(Est_lk.get('est.plugin.vec', est_debias_vec*0)).ravel()
                    # pick first element
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
            #self.var_loading_list = var_loading_list
            #self.var_loading_mat = var_loading_mat

            ### Fitting DRO regression ###
            ## optimized weight vector
            self.weight_ = self.opt_weight(self.Gamma_debias,report_reward=False)['weight']
            ## DRO regression coefficients
            self.loading_coef_ = np.sum([self.Points_info[l]['est_debias_vec'] * self.weight_[l] for l in range(self.L)], axis=0)  # Shape: (n_loading,)


            self.parameters = {
                    'loading_coef_': self.loading_coef_,
                    'weight_': self.weight_
                }

    # ======================================================================= #
    # =================== Prediction  ======================================= #
    # ======================================================================= #
        def predict(self):
            """
            Predict using the fitted DRO regression model.

            Returns
            -------
            np.ndarray
                Predicted values for the target domain features, shape (n0,).
            """
            plugin_weight = self.opt_weight(self.Gamma_plugin,report_reward=False)['weight']
            plugin_coef = np.sum([self.fits_info[l]['beta_init'] * plugin_weight[l] for l in range(self.L)], axis=0)
            pred = self.X0 @ plugin_coef
            return pred

    # ======================================================================= #
    # =================== Compute CIs ======================================= #
    # ======================================================================= #
        def infer(self, M=500, alpha=0.05, alpha_thres=0.01):        
            """
            Placeholder for inference method.
            Perform resampling-based inference to compute confidence intervals for the loading coefficients.
            Parameters
            ----------
            delta : float
                Ridge penalty level, non-positive (default is 0).
            M : int
                Number of resampling iterations (default is 500).
            alpha : float   
                Significance level for confidence intervals (default is 0.05).
            alpha_thres : float
                Significance level for thresholding in resampling (default is 0.01).
            Returns
            -------
            CI_U : np.ndarray
                Confidence intervals for loading coefficients, shape (n_loading, 2).
            Notes
            -----
            This method computes confidence intervals for the loading coefficients using a resampling approach. 
            """
            if not hasattr(self, 'loading_coef_'):
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
                gen_sol = self.opt_weight(self.Gamma_debias, report_reward=False)
                gen_weight_mat[g, :] = gen_sol["weight"]

            ### Constructing CIs ###
            CIs = np.zeros((n_loading, 2))
            for k in range(n_loading):
                loading_coef_0 = np.asarray([self.Points_info[l]['est_debias_vec'][k] for l in range(self.L)])
                gen_loading_coef_ = (gen_weight_mat @ loading_coef_0).reshape(-1)  # Shape: (M,)
                ses = np.asarray([self.Points_info[l]['se_vec'][k] for l in range(self.L)])  # Standard errors for each source domain, shape (L,)
                gen_se = (gen_weight_mat @ ses).reshape(-1)  # Shape: (M,)
                    # Compute confidence intervals
                z_alpha = norm.ppf(1 - alpha / 2)
                gen_CIs_lb = gen_loading_coef_ - z_alpha * gen_se # Shape: (M,)
                gen_CIs_ub = gen_loading_coef_ + z_alpha * gen_se # Shape: (M,)
                CIs[k, 0] = np.min(gen_CIs_lb)
                CIs[k, 1] = np.max(gen_CIs_ub)  
            self.CI_U = CIs


        

    # ========================================================== #    
    # =================== Internal Functions =================== #
    # ========================================================== #

        def opt_weight(self, Gamma, report_reward=False):
            """
            Compute Ridge-type weight vector using cvxpy.
            
            Parameters
            ----------
            Gamma : ndarray (L, L)
                Regression covariance matrix
            delta : float
                Ridge penalty level, non-positive
            report_reward : bool
                Whether to compute penalized reward
            
            Returns
            -------
            dict : 
                'weight' : ndarray (L,)
                'reward' : float (if report_reward=True)
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
            
            if report_reward:
                v2 = cp.Variable(L, nonneg=True)
                objective2 = cp.Minimize(
                    2 * v2.T @ Gamma_positive @ opt_weight
                    - opt_weight.T @ Gamma_positive @ opt_weight
                )
                constraints2 = [cp.sum(v2) == 1]
                prob2 = cp.Problem(objective2, constraints2)
                prob2.solve()
                
                return {
                    "weight": opt_weight,
                    "reward": prob2.value
                }
            else:
                return {"weight": opt_weight}


        def index_map(self, l, k):
            """
            Maps (l, k) with l >= k into vectorized index of upper-triangular part (column-wise),
            matching R: index.map(L, l, k) = (2L - k)(k - 1)/2 + l

            Parameters:
            - L: int, size of the matrix
            - l: int, row index (0-based)
            - k: int, column index (0-based)

            Returns:
            - int: vectorized index
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

                            dev_l1 = self.fits_info[l1]['dev']
                            dev_k1 = self.fits_info[k1]['dev']
                        

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

            Parameters
            ----------
            gen_mu : array-like
                Mean vector of the distribution.
            gen_Cov : array-like
                Covariance matrix of the distribution.
            gen_size : int, optional
                Number of samples to generate. Default is 500.
            threshold : int, optional
                Type of thresholding:
                0 -> coordinate-wise (normal) threshold,
                1 -> chi-square threshold,
                2 -> no threshold (standard MVN sampling).
            alpha_thres : float, optional
                Significance level for thresholding. Default is 0.01.

            Returns
            -------
            gen_samples : np.ndarray
                Generated samples of shape (gen_size, gen_dim).
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




