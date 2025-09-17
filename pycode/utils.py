import numpy as np
import cvxpy as cp
import random
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import xgboost as xgb
from Kden import KernelRatioEstimatorKLIEP

def compute_softmax(x):
    """Compute softmax probabilities for multi-class classification.
    
    Args:
        x: Input matrix of shape (n_samples, num_class-1)
        
    Returns:
        probas: Probability matrix of shape (n_samples, num_class-1)
    """

    # Add zero column for the first class
    x_aug = np.hstack([np.zeros((x.shape[0], 1)), x])
    x_max = np.max(x_aug, axis=1, keepdims=True)
    exp_x = np.exp(x_aug - x_max)
    return exp_x[:, 1:] / np.sum(exp_x, axis=1, keepdims=True)

# --- helpers: classifier proba -> density ratio (with prior correction) ---
def _proba_to_ratio(p, ns, nt):
    """
    Convert p(y=1|x) from a classifier trained on [source vs target] mixture
    into w(x) = p_t(x)/p_s(x), correcting for mixture priors.

    odds = p/(1-p) = (pi_t/pi_s)*w   =>   w = (pi_s/pi_t)*odds
    where pi_s = ns/(ns+nt), pi_t = nt/(ns+nt).
    """
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    pi_s = ns / (ns + nt)
    pi_t = nt / (ns + nt)
    odds = p / (1.0 - p)
    return (pi_s / pi_t) * odds

def fit_fw(models, X_list, y_list, X0):
    """
    Fit the probability and density ratio models.
    
    Args:
        models: An instance of UtilModels class.
        X_list: List of feature matrices for each source domain.
        y_list: List of label arrays for each source domain.
        X0: Feature matrix for the target domain.
    Returns:
        fX_list: List of predicted probabilities for each source domain.
        fX0_list: List of predicted probabilities for the target domain.
        wX_list: List of density ratio estimates for each source domain.
    """
    # --------------- Compute probas ---------------- #
    # Fit the models
    fX_list = []
    fX0_list = []
    L = len(X_list)
    for l in range(L):
        models.fit_f(X_list[l], y_list[l])
        fX, fX0 = models.pred_f(X_list[l], X0)
        fX_list.append(fX)
        fX0_list.append(fX0)

    # --------------- Compute density ratios ---------------- #
    wX_list = []
    for l in range(L):
        models.fit_w(X_list[l], X0)
        wX = models.pred_w(X_list[l], X0)
        wX_list.append(wX)


    return fX_list, fX0_list, wX_list
    

class UtilModels:
    def __init__(self, mode='cls', f_learner='linear', w_learner='linear', lambda_val=None, split=False, intercept=False, seed=123, verbose=False): 
        """Initialize the model utility class with parameters.
        Args:
            seed : int, optional
                Random seed for reproducibility.
            mode: str, whether to use 'reg' or 'cls'
            f_learner : str, default='linear'
                ('linear' or 'xgb')
            w_learner : str, default='linear'
                ('linear' or 'xgb' or 'ulsif' or 'rulsif' or 'kliep')
            split : bool, default=False
                Whether to split data for creating independence.
            verbose : bool, default=False
                Verbosity of GridSearchCV.
            lambda_val : float or str, default=None
                Regularization parameter for high-dimensional linear regression.
                If 'CV.min', use the lambda that gives minimum CV error.
                If 'CV', use the largest lambda within 1 standard error of the minimum CV error.
                If None, no regularization is applied.
            intercept : bool, default=False (for regression only)
                Whether to include an intercept term in the regression model.
        """
        self.seed = seed
        np.random.seed(seed)

        self.mode = mode
        self.f_learner = f_learner
        self.w_learner = w_learner
        self.lambda_val = lambda_val
        self.split = split
        self.intercept = intercept
        self.verbose = verbose

    def fit_f(self, X, y):
        """Fit a probability model on the source domain and evaluate it on both the source and target domains.

        Args:
            X : array-like of shape (n_samples, n_features)
                Feature matrix for the source domain.
            y : array-like of shape (n_samples,)
                Class labels for the source domain.
            X0 : array-like of shape (m_samples, n_features)
                Feature matrix for the target domain.

        Returns:
            predX : array-like of shape (n_samples, K)
                Predicted probabilities for the source domain.
            predX0 : array-like of shape (m_samples, K)
                Predicted probabilities for the target domain
        """

        verbose = 0 if not self.verbose else 1

        # ================== Model Definition ================== #
        if self.mode == 'cls':
            self.num_class = len(np.unique(y))
            if self.f_learner == 'linear':
                estimator = LogisticRegression(solver='lbfgs', max_iter=10000)
                param_grid = None
            elif self.f_learner == 'xgb':
                estimator = xgb.XGBClassifier(
                objective='multi:softprob',   # Outputs class probabilities
                num_class=self.num_class,          # Number of classes
                eval_metric='mlogloss',       # Metric for multi-class log loss
                n_jobs=-1,                    # Use all CPU cores
                seed=self.seed
                )
                param_grid = {
                    'learning_rate': [0.05, 0.01],    # Common values
                    'max_depth': [3, 6, 9],                # Control model complexity
                    'subsample': [0.8, 1.0],               # Prevent overfitting
                    'colsample_bytree': [0.8, 1.0],        # Feature subsampling
                }
        elif self.mode == 'reg':
            if self.f_learner == 'linear':
                estimator = LinearRegression()
                param_grid = None
            elif self.f_learner == 'high_d':
                max_iter = 10000
                param_grid = None
                if self.lambda_val is None or self.lambda_val == "CV.min":
                    estimator = LassoCV(cv=10, fit_intercept=self.intercept, max_iter=max_iter)

                elif self.lambda_val == "CV":
                    # Note: scikit-learn's LassoCV does not have lambda.1se directly
                    # We'll approximate by picking alpha with minimum CV error + 1 std
                    estimator = LassoCV(cv=10, fit_intercept=self.intercept, max_iter=max_iter).fit(X, y)
                    mse_path_mean = estimator.mse_path_.mean(axis=1)
                    mse_path_std = estimator.mse_path_.std(axis=1)
                    min_idx = np.argmin(mse_path_mean)
                    # Find largest alpha whose MSE <= min_MSE + std_MSE
                    mse_1se_threshold = mse_path_mean[min_idx] + mse_path_std[min_idx]
                    idx_1se = np.where(mse_path_mean <= mse_1se_threshold)[0][-1]
                    alpha_1se = estimator.alphas_[idx_1se]
                    estimator = Lasso(alpha=alpha_1se, fit_intercept=self.intercept, max_iter=max_iter)

                else:
                    # lambda_val is numeric
                    estimator = Lasso(alpha=self.lambda_val, fit_intercept=self.intercept, max_iter=max_iter)

            elif self.f_learner == 'xgb':
                estimator = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    n_estimators=200,
                    n_jobs=-1,
                    seed=self.seed
                )
                param_grid = {
                'learning_rate': [0.1], #[0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
                'max_depth': [3,6],#[3, 6, 9],          # Maximum depth of a tree
                'subsample': [0.8], #, 1.0],         # Row fraction
                'colsample_bytree': [0.8] # [0.8, 1.0],  # Feature fraction
            }
        else:
            assert False, "mode should be either 'cls' or 'reg'"
        # ================== Model Training / Prediction ================== #
        if self.split:
            # If split is True, split the data into halves A and B
            n = X.shape[0]
            self.indA = np.random.choice(n, int(n/2), replace=False)
            self.indB = np.array(list(set(range(n)) - set(self.indA)))
            XA, yA = X[self.indA], y[self.indA]
            XB, yB = X[self.indB], y[self.indB]
            
            if param_grid is None: # No hyperparameter tuning, for logistic regression
                self.modelA_f = estimator.fit(XA, yA)
                self.modelB_f = estimator.fit(XB, yB)

            else:
                if self.mode == "cls":
                    # Set up 5-fold stratified cross-validation
                    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    gridA = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose
                    )
                    gridB = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose,
                        n_jobs=-1
                    )
                elif self.mode == "reg":
                    kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
                    gridA = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error',  # Use mean squared error for regression
                        verbose=verbose
                    )
                    gridB = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error',  # Use mean squared error for regression
                        verbose=verbose,
                        n_jobs=-1
                    )
                else:
                    assert False, "mode should be either 'cls' or 'reg'"

                gridA.fit(XA, yA)
                gridB.fit(XB, yB)
                self.modelA_f = gridA.best_estimator_
                self.modelB_f = gridB.best_estimator_
                self.proba_params = gridA.best_params_ # Keep only the parameters of the first model
        
        
        else:
            # No split: we just train on the entire dataset
            if param_grid is None: # No hyperparameter tuning, for logistic regression
                self.model_f = estimator.fit(X, y)

            else:
                if self.mode == "cls":
                    # Set up 5-fold stratified cross-validation
                    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    grid = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                    cv=stratified_kfold,
                    scoring='neg_log_loss',        # Use log loss for probability calibration
                    verbose=verbose,
                    n_jobs=-1
                )
                elif self.mode == "reg":
                    kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
                    grid = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error',  # Use mean squared error for regression
                        verbose=verbose
                    )
                else:
                    assert False, "mode should be either 'cls' or 'reg'"

                grid.fit(X, y)
                self.model_f = grid.best_estimator_
                self.proba_params = grid.best_params_


    def pred_f(self, X, X0):
        """
        Generate predictions for the source and target domains.

        Args:
            X : array-like of shape (n_samples, n_features)
                Feature matrix for the source domain.
            X0 : array-like of shape (m_samples, n_features)
                Feature matrix for the target domain.

        Returns:
            predX : array-like of shape (n_samples, n_classes)
                Predicted probabilities for the source domain.
            predX0 : array-like of shape (m_samples, n_classes)
                Predicted probabilities for the target domain.
        """
        if self.mode == 'cls':
            if self.split:
                modelA = self.modelA_f
                modelB = self.modelB_f
                XA = X[self.indA]
                XB = X[self.indB]
                X0 = X0

                # Generate predictions for the source and target domains
                predAB = modelA.predict_proba(XB) # Predictions of hatPA on XB
                predBA = modelB.predict_proba(XA) # Predictions of hatPB on XA
                predX0 = 0.5 * modelA.predict_proba(X0) + 0.5 * modelB.predict_proba(X0)
                    
                predX = np.zeros((X.shape[0], self.num_class))
                predX[self.indA, :] = predBA
                predX[self.indB, :] = predAB


            else:
                # Generate predictions for the source and target domains
                predX = self.model_f.predict_proba(X)
                predX0 = self.model_f.predict_proba(X0)

            # Clip the predicted probabilities to avoid numerical instability
            predX = np.clip(predX, 1e-3, 1-1e-3)
            predX0 = np.clip(predX0, 1e-3, 1-1e-3)
        elif self.mode == 'reg':
            if self.split:
                modelA = self.modelA_f
                modelB = self.modelB_f
                XA = X[self.indA]
                XB = X[self.indB]
                X0 = X0

                # Generate predictions for the source and target domains
                predAB = modelA.predict(XB)
                predBA = modelB.predict(XA)
                predX0 = 0.5 * modelA.predict(X0) + 0.5 * modelB.predict(X0)

                predX = np.zeros(X.shape[0])    
                predX[self.indA] = predBA
                predX[self.indB] = predAB

            else:
                # Generate predictions for the source and target domains
                predX = self.model_f.predict(X)
                predX0 = self.model_f.predict(X0)
        else:
            assert False, "mode should be either 'cls' or 'reg'"
        

        return predX, predX0
    
    



    def fit_w(self, X, X0):
        """Fit a density-ratio model (no return; models stored on `self`)."""
        verbose = 0 if not getattr(self, "verbose", False) else 1
        X = np.asarray(X); X0 = np.asarray(X0)

        # ----------------- Model Definition ----------------- #
        if self.w_learner == 'linear':
            estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
            param_grid = None

        elif self.w_learner == 'xgb':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_jobs=-1,
                seed=self.seed
            )
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8]
            }


        elif self.w_learner == 'kliep':
            estimator = KernelRatioEstimatorKLIEP(
                sigma=getattr(self, 'w_sigma', None),
                n_centers=getattr(self, 'w_n_centers', 200),
                center_target=True,
                lr=getattr(self, 'w_lr', 0.1),
                max_iter=getattr(self, 'w_max_iter', 500),
                tol=getattr(self, 'w_tol', 1e-6),
                random_state=self.seed,
                clip_min=1e-12
            )
            #estimator = DensityRatioEstimator()
            param_grid = None

        else:
            raise ValueError(f"Unknown w_learner: {self.w_learner}")

        # ----------------- Train (with or without split) ----------------- #
        if getattr(self, "split", False):
            if not hasattr(self, "indA") or not hasattr(self, "indB"):
                raise ValueError("Cross-fitting requires self.indA and self.indB.")
            indA = np.asarray(self.indA); indB = np.asarray(self.indB)
            XA, XB = X[indA], X[indB]

            if self.w_learner in ('linear', 'xgb'):
                XA_concat = np.vstack((XA, X0))
                XB_concat = np.vstack((XB, X0))
                yA_concat = np.concatenate((np.zeros(XA.shape[0]), np.ones(X0.shape[0])))
                yB_concat = np.concatenate((np.zeros(XB.shape[0]), np.ones(X0.shape[0])))

                if param_grid is None:
                    self.modelA_w = estimator.fit(XA_concat, yA_concat)
                    self.modelB_w = estimator.fit(XB_concat, yB_concat)
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    gridA = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                                        scoring='neg_log_loss', verbose=verbose, n_jobs=-1)
                    gridB = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                                        scoring='neg_log_loss', verbose=verbose, n_jobs=-1)
                    gridA.fit(XA_concat, yA_concat)
                    gridB.fit(XB_concat, yB_concat)
                    self.modelA_w = gridA.best_estimator_
                    self.modelB_w = gridB.best_estimator_
                    self.density_params = gridA.best_params_

            else:
                # kernel learners: fit on (XA, X0) and (XB, X0)
                est_params = estimator.get_params()
                estA = estimator.__class__(**est_params)
                estB = estimator.__class__(**est_params)
                #estA = estB = estimator
                self.modelA_w = estA.fit(XA, X0)
                self.modelB_w = estB.fit(XB, X0)
                self.density_params = est_params
                #self.density_params = None
        else:
            if self.w_learner in ('linear', 'xgb'):
                X_concat = np.vstack((X, X0))
                y_concat = np.concatenate((np.zeros(X.shape[0]), np.ones(X0.shape[0])))
                if param_grid is None:
                    self.model_w = estimator.fit(X_concat, y_concat)
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                                        scoring='neg_log_loss', verbose=verbose, n_jobs=-1)
                    grid.fit(X_concat, y_concat)
                    self.model_w = grid.best_estimator_
                    self.density_params = grid.best_params_
            else:
                self.model_w = estimator.fit(X, X0)
                self.density_params = estimator.get_params()

        self.density_learner = self.w_learner  # record choice


    def pred_w(self, X, X0):
        """
        Generate density ratio estimates for the source domain.

        Args:
            X  : (n_samples, n_features) source
            X0 : (m_samples, n_features) target
        Returns:
            wX : (n_samples,) estimated w(x) = p_t(x)/p_s(x)
        """
        X = np.asarray(X); X0 = np.asarray(X0)
        clip_lo = getattr(self, "w_clip_lo", 1e-3)
        clip_hi = getattr(self, "w_clip_hi", 1e3)

        if getattr(self, "split", False):
            if not hasattr(self, "indA") or not hasattr(self, "indB"):
                raise ValueError("Cross-fitting requires self.indA and self.indB.")
            indA = np.asarray(self.indA); indB = np.asarray(self.indB)
            XA, XB = X[indA], X[indB]
            nt = X0.shape[0]

            if self.density_learner in ('linear', 'xgb'):
                # cross-predict probabilities, convert with per-fold priors
                proba_AB = self.modelA_w.predict_proba(XB)[:, 1]  # modelA on XB
                proba_BA = self.modelB_w.predict_proba(XA)[:, 1]  # modelB on XA
                wB = _proba_to_ratio(proba_AB, ns=XB.shape[0], nt=nt)
                wA = _proba_to_ratio(proba_BA, ns=XA.shape[0], nt=nt)
            else:
                # kernel: predict direct ratios cross-wise
                wA = self.modelB_w.predict(XA)
                wB = self.modelA_w.predict(XB)

            wX = np.empty(X.shape[0], dtype=float)
            wX[indA] = wA
            wX[indB] = wB

        else:
            if self.density_learner in ('linear', 'xgb'):
                proba = self.model_w.predict_proba(X)[:, 1]
                wX = _proba_to_ratio(proba, ns=X.shape[0], nt=X0.shape[0])
            else:
                wX = self.model_w.predict(X)

        # final clipping for numerical stability
        wX = np.clip(wX, clip_lo, clip_hi)
        return wX



class BiasCorrection:
    def __init__(self, lam=None, intercept=False, loading_intercept=False, verbose=False):
        self.lam = lam
        self.intercept = intercept
        self.loading_intercept = loading_intercept
        self.verbose = verbose

    def LF(self, X, y, loading_mat, beta_init=None, xi=None):


        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        loading_mat = np.asarray(loading_mat, dtype=float) # n_loading * p

        loading_include_intercept = (X.shape[1] == (loading_mat.shape[0] - 1))

        # --- Preparation ---
        # Center X
        X_means = X.mean(axis=0)
        X = X - X_means  # center, no scaling



        # Initial lasso estimator
        if beta_init is None:
            UM = UtilModels(mode='reg', f_learner='high_d', lambda_val=self.lam, split=False, verbose=self.verbose)
            UM.fit_f(X, y)
            Umodel = UM.model_f
            beta_init = np.concatenate(([Umodel.intercept_], Umodel.coef_)) if self.intercept else Umodel.coef_
        beta_init = np.asarray(beta_init).ravel()
        sparsity = np.sum(np.abs(beta_init) > 1e-4)

        # Prepare X for model fitting
        if self.intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        n, p = X.shape
        pred = (X @ beta_init).ravel()
        deriv = np.ones_like(pred)
        weight = np.ones_like(pred)
        cond_var = self.cond_var_fun(pred, y, sparsity).ravel()

        # --- Filter observations ---
        idx = np.ones(n, dtype=bool)


        X_filter = X[idx, :]
        y_filter = y[idx]
        weight_filter = weight[idx]
        deriv_filter = deriv[idx]
        n_filter = X_filter.shape[0]

        # --- Initialize result storage ---
        n_loading = loading_mat.shape[0]
        est_plugin_vec = np.full(n_loading, np.nan)
        est_debias_vec = np.full(n_loading, np.nan)
        se_vec = np.full(n_loading, np.nan)
        proj_mat = np.full((p, n_loading), np.nan)

        for i_loading in range(n_loading):
            if self.verbose:
                print(f"---> Computing for loading ({i_loading+1}/{n_loading})...")

            # Adjust loading
            loading = loading_mat[i_loading, :].copy()
            if not loading_include_intercept:
                if self.intercept:
                    if self.loading_intercept:
                        loading = loading - X_means
                        loading = np.concatenate([[1], loading])
                    else:
                        loading = np.concatenate([[0], loading])

            loading_norm = np.sqrt(np.sum(loading ** 2))

            # Correction direction
            direction = self.compute_direction(loading, X_filter, weight_filter, deriv_filter, xi, self.verbose)

            # Bias correction
            est_plugin = np.sum(beta_init * loading)
            correction = np.mean((weight * (y - pred))[:, None] * X @ direction)
            est_debias = est_plugin + correction * loading_norm

            # Compute SE
            V = np.sum(((np.sqrt(weight ** 2 * cond_var)[:, None] * X) @ direction) ** 2) / n * loading_norm ** 2
            se = np.sqrt(V / n)

            # Store results
            est_plugin_vec[i_loading] = est_plugin
            est_debias_vec[i_loading] = est_debias
            se_vec[i_loading] = se
            proj_mat[:, i_loading] = direction * loading_norm

        return {
            "est_plugin_vec": est_plugin_vec,
            "est_debias_vec": est_debias_vec,
            "se_vec": se_vec,
            "proj_mat": proj_mat
        }



    def get_mode(self, v):
        counts = Counter(v)
        if all(c == 1 for c in counts.values()):
            return float(np.median(v))
        else:
            return float(max(counts, key=counts.get))
        
    def cond_var_fun(self, pred, y=None, sparsity=None):
            if y is None:
                raise ValueError("y must be provided for linear model cond_var_fun.")
            n = len(y)
            denom = max(0.7 * n, n - sparsity if sparsity is not None else n)
            sigma_sq = np.sum((y - pred) ** 2) / denom
            return np.repeat(sigma_sq, n)



    def direction_search_tuning(self,X, loading, weight, deriv, resol=1.5, maxiter=10):
        p = X.shape[1]
        n = X.shape[0]
        xi = np.sqrt(2.01 * np.log(p) / n)
        loading = np.asarray(loading).reshape(-1)
        loading_norm = np.linalg.norm(loading)
        opt_sol = np.zeros(p + 1)

        H = np.column_stack((loading / loading_norm, np.eye(p)))
        adj_XH = np.sqrt(weight)[:, None] * np.sqrt(deriv)[:, None] * (X @ H)

        v = cp.Variable(p + 1)

        # First iteration to decide incr
        obj = (1/4) * cp.sum_squares(adj_XH @ v) / n \
            + cp.sum((loading / loading_norm) @ (H @ v)) \
            + xi * cp.norm1(v)
        prob = cp.Problem(cp.Minimize(obj))
        result = prob.solve()
        status = prob.status

        if status == "optimal":
            incr = -1
            v_opt = v.value
        else:
            incr = 1

        # Search loop
        iter_count = 1
        while iter_count <= maxiter:
            laststatus = status
            xi *= resol ** incr
            obj = (1/4) * cp.sum_squares(adj_XH @ v) / n \
                + cp.sum((loading / loading_norm) @ (H @ v)) \
                + xi * cp.norm1(v)
            prob = cp.Problem(cp.Minimize(obj))
            result = prob.solve()
            status = prob.status

            if incr == -1:
                if status == "optimal":
                    v_opt = v.value
                    iter_count += 1
                    continue
                else:
                    step = iter_count - 1
                    break
            if incr == 1:
                if status != "optimal":
                    iter_count += 1
                    continue
                else:
                    step = iter_count
                    v_opt = v.value
                    break
        else:
            step = maxiter

        direction = -(0.5) * (v_opt[1:] + v_opt[0] * loading / loading_norm)
        return {
            "proj": direction,
            "step": step,
            "incr": incr,
            "laststatus": laststatus,
            "curstatus": status,
            "xi": xi
        }


    def direction_fixed_tuning(self,X, loading, weight, deriv, xi=None, resol=1.5, step=3, incr=-1):
        p = X.shape[1]
        n = X.shape[0]
        loading = np.asarray(loading).reshape(-1)

        if xi is None:
            xi = np.sqrt(2.01 * np.log(p) / n)
            xi = xi * (resol ** (incr * step))

        loading_norm = np.linalg.norm(loading)
        H = np.column_stack((loading / loading_norm, np.eye(p)))
        
        v = cp.Variable(p + 1)
        adj_XH = np.sqrt(weight)[:, None] * np.sqrt(deriv)[:, None] * (X @ H)
        
        obj = (1/4) * cp.sum_squares(adj_XH @ v) / n \
            + cp.sum((loading / loading_norm) @ (H @ v)) \
            + xi * cp.norm1(v)

        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        
        opt_sol = v.value
        status = prob.status
        direction = (-0.5) * (opt_sol[1:] + opt_sol[0] * loading / loading_norm)
        
        return {
            "proj": direction,
            "status": status,
            "xi": xi
        }




    def compute_direction(self,loading, X, weight, deriv, xi=None, verbose=False):
        n, p = X.shape
        loading = np.asarray(loading).reshape(-1)
        loading_norm = np.linalg.norm(loading)

        if loading_norm <= 1e-5:
            if verbose:
                print("Loading norm too small, setting proj direction to zeros.")
            direction = np.zeros_like(loading)
        else:
            if n >= 6 * p:
                # Low-dimensional case
                temp = (np.sqrt(weight * deriv)[:, None]) * X
                Sigma_hat = temp.T @ temp / n
                direction = np.linalg.solve(Sigma_hat, loading) / loading_norm
            else:
                direction_alter = False
                try:
                    if xi is None:
                        # xi not specified
                        if n >= 0.9 * p:
                            step_vec = []
                            incr_vec = []
                            for _ in range(3):
                                index_sel = random.sample(range(n), round(0.9 * p))
                                Direction_Est_temp = self.direction_search_tuning(
                                    X[index_sel, :],
                                    loading,
                                    weight=weight[index_sel],
                                    deriv=deriv[index_sel]
                                )
                                step_vec.append(Direction_Est_temp["step"])
                                incr_vec.append(Direction_Est_temp["incr"])
                            step = self.get_mode(step_vec)
                            incr = self.get_mode(incr_vec)

                            Direction_Est = self.direction_fixed_tuning(
                                X, loading, weight=weight, deriv=deriv, step=step, incr=incr
                            )
                            while Direction_Est["status"] != "optimal":
                                step += incr
                                Direction_Est = self.direction_fixed_tuning(
                                    X, loading, weight=weight, deriv=deriv, step=step, incr=incr
                                )
                            if verbose:
                                print(f"The projection direction is identified at xi = {Direction_Est['xi']:.6f} at step = {step}")
                        else:
                            Direction_Est = self.direction_search_tuning(
                                X, loading, weight=weight, deriv=deriv
                            )
                            if verbose:
                                print(f"The projection direction is identified at xi = {Direction_Est['xi']:.6f} at step = {Direction_Est['step']}")
                    else:
                        # xi specified
                        Direction_Est = self.direction_fixed_tuning(
                            X, loading, weight=weight, deriv=deriv, xi=xi
                        )
                        while Direction_Est["status"] != "optimal":
                            xi *= 1.5
                            Direction_Est = self.direction_fixed_tuning(
                                X, loading, weight=weight, deriv=deriv, xi=xi
                            )
                            if verbose:
                                print(f"The projection direction is identified at xi = {Direction_Est['xi']:.6f}")
                    direction = Direction_Est["proj"]

                except Exception as e:
                    print("Caught an error using cvxpy! Alternative method is applied for proj direction.")
                    print(e)
                    direction_alter = True

                if direction_alter:
                    temp = (np.sqrt(weight * deriv)[:, None]) * X
                    Sigma_hat = temp.T @ temp / n
                    #print('shape of Sigma_hat:', Sigma_hat.shape)
                    Sigma_hat_inv = np.diag(1 / np.diag(Sigma_hat))
                    direction = Sigma_hat_inv @ loading / loading_norm

        return direction









