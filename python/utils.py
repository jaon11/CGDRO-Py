import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import xgboost as xgb

### Cross-entropy
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

class UtilModels:
    def __init__(self, proba_params=None, density_params=None, seed=None):
        """Initialize the model utility class with parameters.
        Args:
            proba_params : dict, optional
                Pre-tuned parameters for probability model to bypass grid search.
            density_params : dict, optional
                Pre-tuned parameters for density model to bypass grid search.
            seed : int, optional
                Random seed for reproducibility.
        """
        self.proba_params = proba_params
        self.density_params = density_params
        self.seed = seed
        np.random.seed(seed)
    
    def compute_proba(self, X, y, X0, prob_learner='linear', split=False, verbose=False):
        """Fit a probability model on the source domain and evaluate it on both the source and target domains.

        Args:
            X : array-like of shape (n_samples, n_features)
                Feature matrix for the source domain.
            y : array-like of shape (n_samples,)
                Class labels for the source domain.
            X0 : array-like of shape (m_samples, n_features)
                Feature matrix for the target domain.
            prob_learner : str, default='linear'
                ('linear' or 'xgb)
            split : bool, default=False
                Whether to split data for creating independence.
            verbose : bool, default=False
                Verbosity of GridSearchCV.

        Returns:
            predX : array-like of shape (n_samples, K)
                Predicted probabilities for the source domain.
            predX0 : array-like of shape (m_samples, K)
                Predicted probabilities for the target domain
        """     
        num_class = len(np.unique(y))
        verbose = 0 if not verbose else 1
        
        # ================== Model Definition ================== #
        if prob_learner == 'linear':
            estimator = LogisticRegression(solver='lbfgs')
            param_grid = None
        elif prob_learner == 'xgb':
            estimator = xgb.XGBClassifier(
                objective='multi:softprob',   # Outputs class probabilities
                num_class=num_class,          # Number of classes
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
        
        # ================== Model Training / Prediction ================== #
        if split:
            # If split is True, split the data into halves A and B
            n = X.shape[0]
            indA = np.random.choice(n, int(n/2), replace=False)
            indB = np.array(list(set(range(n)) - set(indA)))
            XA, yA = X[indA], y[indA]
            XB, yB = X[indB], y[indB]
            
            if param_grid is None: # No hyperparameter tuning, for logistic regression
                modelA = estimator.fit(XA, yA)
                modelB = estimator.fit(XB, yB)

            else:
                if self.proba_params is not None:
                    estimator.set_params(**self.proba_params)
                    modelA = estimator.fit(XA, yA)
                    modelB = estimator.fit(XB, yB)
                else:
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
                    gridA.fit(XA, yA)
                    gridB.fit(XB, yB)
                    modelA = gridA.best_estimator_
                    modelB = gridB.best_estimator_
                    self.proba_params = gridA.best_params_ # Keep only the parameters of the first model
            
            # Generate predictions for the source and target domains
            predAB = modelA.predict_proba(XB) # Predictions of hatPA on XB
            predBA = modelB.predict_proba(XA) # Predictions of hatPB on XA
            predX0 = 0.5 * modelA.predict_proba(X0) + 0.5 * modelB.predict_proba(X0)
                
            predX = np.zeros((X.shape[0], num_class))
            predX[indA, :] = predBA
            predX[indB, :] = predAB
        
        else:
            # No split: we just train on the entire dataset
            if param_grid is None: # No hyperparameter tuning, for logistic regression
                model = estimator.fit(X, y)

            else:
                if self.proba_params is not None:
                    estimator.set_params(**self.proba_params)
                    model = estimator.fit(X, y)
                
                else:
                    # Set up 5-fold stratified cross-validation
                    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    grid = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose,
                        n_jobs=-1
                    )
                    grid.fit(X, y)
                    model = grid.best_estimator_
                    self.proba_params = grid.best_params_
            
            # Generate predictions for the source and target domains
            predX = model.predict_proba(X)
            predX0 = model.predict_proba(X0)
        
        # Clip the predicted probabilities to avoid numerical instability
        predX = np.clip(predX, 1e-3, 1-1e-3)
        predX0 = np.clip(predX0, 1e-3, 1-1e-3)
        
        return predX, predX0
    
    def compute_density(self, X, X0, density_learner='linear', split=False, verbose=False):
        """Fit a density model on the source domain and evaluate it on source domains.

        Args:
            X : array-like of shape (n_samples, n_features)
                Feature matrix for the source domain.
            X0 : array-like of shape (m_samples, n_features)
                Feature matrix for the target domain.
            density_learner : str, default='linear'
            Learner type ('linear' or 'xgb').
            split : bool, default=False
                Whether to split data for creating independence.
            verbose : bool, default=False
                Verbosity of GridSearchCV.

        Returns:
            omegaX : array-like of shape (n_samples,)
                Density ratio estimates for source domain.   
        """
        verbose = 0 if not verbose else 1
        
        # ================== Model Definition ================== #
        if density_learner == 'linear':
            estimator = LogisticRegression(solver='lbfgs')
            param_grid = None
        elif density_learner == 'xgb':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',  # Binary classification objective
                eval_metric='logloss',        # Metric for binary classification
                n_jobs=-1,                    # Use all CPU cores
                seed=self.seed
            )
            param_grid = {
                'learning_rate': [0.05, 0.01],    # Common values
                'max_depth': [3, 6, 9],                # Control model complexity
                'subsample': [0.8, 1.0],               # Prevent overfitting
                'colsample_bytree': [0.8, 1.0],        # Feature subsampling
            }
            
        # ================== Model Training / Prediction ================== #
        if split:
            # If split is True, split the source data into halves A and B
            n = X.shape[0]
            indA = np.random.choice(n, int(n/2), replace=False)
            indB = np.array(list(set(range(n)) - set(indA)))
            XA_concat = np.vstack((X[indA], X0))
            XB_concat = np.vstack((X[indB], X0))
            yA_concat = np.concatenate((np.zeros(X[indA].shape[0]), np.ones(X0.shape[0])))
            yB_concat = np.concatenate((np.zeros(X[indB].shape[0]), np.ones(X0.shape[0])))

            if param_grid is None: # No hyperparameter tuning, for logistic regression
                modelA = estimator.fit(XA_concat, yA_concat)
                modelB = estimator.fit(XB_concat, yB_concat)
            else:
                if self.density_params is not None:
                    # Use pre-tuned parameters
                    estimator.set_params(**self.density_params)
                    modelA = estimator.fit(XA_concat, yA_concat)
                    modelB = estimator.fit(XB_concat, yB_concat)
                    
                else:
                    # Set up 5-fold stratified cross-validation
                    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    gridA = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose,
                        n_jobs=-1
                    )
                    gridB = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose,
                        n_jobs=-1
                    )
                    gridA.fit(XA_concat, yA_concat)
                    gridB.fit(XB_concat, yB_concat)
                    modelA = gridA.best_estimator_
                    modelB = gridB.best_estimator_
                    self.density_params = gridA.best_params_ # Keep only the parameters of the first model
                
            prob_ratioAB = modelA.predict_proba(X[indB])[:, 1] / modelA.predict_proba(X[indB])[:, 0]
            prob_ratioBA = modelB.predict_proba(X[indA])[:, 1] / modelB.predict_proba(X[indA])[:, 0]
            omegaB = prob_ratioAB * (X[indA].shape[0] / X0.shape[0])
            omegaA = prob_ratioBA * (X[indB].shape[0] / X0.shape[0])
            
            omegaX = np.zeros(X.shape[0])
            omegaX[indA] = omegaA
            omegaX[indB] = omegaB
        
        else:
            # No split: we just train on the entire source dataset
            X_concat = np.vstack((X, X0))
            y_concat = np.concatenate((np.zeros(X.shape[0]), np.ones(X0.shape[0])))
            if param_grid is None: # No hyperparameter tuning, for logistic regression
                model = estimator.fit(X_concat, y_concat)
            else:
                if self.density_params is not None:
                    # Use pre-tuned parameters
                    estimator.set_params(**self.density_params)
                    model = estimator.fit(X_concat, y_concat)
                else:
                    # Set up 5-fold stratified cross-validation
                    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                    grid = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=stratified_kfold,
                        scoring='neg_log_loss',          # Use log loss for probability calibration
                        verbose=verbose,
                        n_jobs=-1
                    )
                    grid.fit(X_concat, y_concat)
                    model = grid.best_estimator_
                    self.density_params = grid.best_params_
            
            prob_ratio = model.predict_proba(X)[:, 1] / model.predict_proba(X)[:, 0]
            omegaX = prob_ratio * (X.shape[0] / X0.shape[0])
        
        # Clip the density ratio omegaX to avoid numerical instability
        omegaX = np.clip(omegaX, 1e-3, 1e3)    
        return omegaX


### DRoL
def reward(Y, Y_pred):
    return np.mean(Y**2 - (Y - Y_pred) ** 2)

class OutcomeModel:
    def __init__(self, learner='linear', params=None):
        """Initialize the OutcomeModel class with parameters (if applicable).
        """
        self.learner = learner
        self.params = params
        self.model = None
    
    def fit(self, X, Y, sample_weight=None, verbose=0, seed=None):
        """Fit the outcome model.
        
        Args:
            X (np.ndarray): features
            Y (np.ndarray): outcomes
        """
        if sample_weight is None:
            sample_weight = np.ones_like(Y)
 
        if self.learner == 'linear':
            self.model = LinearRegression().fit(X, Y, sample_weight=sample_weight)
            param_grid = None
        if self.learner == 'xgb':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=200,
                n_jobs=-1
            )
            param_grid = {
                'learning_rate': [0.1], #[0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
                'max_depth': [3,6],#[3, 6, 9],          # Maximum depth of a tree
                'subsample': [0.8], #, 1.0],         # Row fraction
                'colsample_bytree': [0.8] # [0.8, 1.0],  # Feature fraction
            }
            if self.params is not None:
                self.model.set_params(**self.params)
                self.model.fit(X, Y, sample_weight=sample_weight)
            else:
                kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
                grid = GridSearchCV(estimator = self.model, 
                                    param_grid = param_grid, 
                                    cv=kfold,
                                    scoring='neg_mean_squared_error',
                                    n_jobs=-1,
                                    verbose=verbose)
                grid.fit(X, Y, sample_weight=sample_weight)
                self.model = grid.best_estimator_
                self.params = grid.best_params_
    
    def predict(self, X):
        return self.model.predict(X)
    

class DensityModel:
    def __init__(self, learner='linear', params=None):
        """Initialize the DensityModel class with parameters (if applicable).
        """
        self.learner = learner
        self.params = params
        self.model = None
        self.sample_ratio = None
        
    def fit(self, X, X_target, verbose=0, seed=None):
        """Fit the density model.
        
        Args:
            X (np.ndarray): features
            X_target (np.ndarray): target features
        """
        self.sample_ratio = X.shape[0] / X_target.shape[0]
        X_concat = np.vstack((X, X_target))
        Y_concat = np.concatenate((np.zeros(X.shape[0]), np.ones(X_target.shape[0])))
        if self.learner == 'logistic':
            self.model = LogisticRegression(solver='lbfgs').fit(X_concat, Y_concat)
            param_grid = None
        if self.learner == 'xgb':
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=200,
                n_jobs=-1
            )
            param_grid = {
                'learning_rate': [0.1], #[0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
                'max_depth': [3,6],#[3, 6, 9],          # Maximum depth of a tree
                'subsample': [0.8], #, 1.0],         # Row fraction
                'colsample_bytree': [0.8] # [0.8, 1.0],  # Feature fraction
            }
            # param_grid = {
            #     'learning_rate': [0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
            #     'max_depth': [3, 6, 9],          # Maximum depth of a tree
            #     'subsample': [0.8, 1.0],         # Row fraction
            #     'colsample_bytree': [0.8, 1.0],  # Feature fraction
            # }
            if self.params is not None:
                self.model.set_params(**self.params)
                self.model.fit(X_concat, Y_concat)
            else:
                stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                grid = GridSearchCV(estimator = self.model, 
                                    param_grid = param_grid, 
                                    cv=stratified_kfold,
                                    scoring='neg_log_loss',
                                    n_jobs=-1,
                                    verbose=verbose)
                grid.fit(X_concat, Y_concat)
                self.model = grid.best_estimator_
                self.params = grid.best_params_
        
    
    def predict(self, X):
        proba_ratio = self.model.predict_proba(X)[:, 1] / self.model.predict_proba(X)[:, 0]
        omega = proba_ratio * self.sample_ratio
        return omega
