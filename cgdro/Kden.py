import numpy as np
from copy import deepcopy

# ---------- Shared RBF kernel utilities ----------
def _pairwise_rbf(X, C, sigma):
    # K_{ij} = exp(-||X_i - C_j||^2/(2 sigma^2))
    X = np.asarray(X, float); C = np.asarray(C, float)
    X2 = np.sum(X**2, axis=1, keepdims=True)
    C2 = np.sum(C**2, axis=1, keepdims=True).T
    d2 = X2 + C2 - 2.0 * X @ C.T
    return np.exp(-d2 / (2.0 * sigma * sigma))

def _median_heuristic(Xs, Xt, max_samples=1000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.vstack([Xs, Xt])
    if X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], size=max_samples, replace=False)]
    X2 = np.sum(X**2, axis=1, keepdims=True)
    d2 = X2 + X2.T - 2.0 * (X @ X.T)
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices_from(d, k=1)
    med = np.median(d[iu])
    return max(med, 1e-6)


# ---------- KLIEP ----------
class KernelRatioEstimatorKLIEP:
    def __init__(self, sigma=None, n_centers=200, center_target=True, lr=0.1, max_iter=500,
                 tol=1e-6, random_state=None, clip_min=1e-12):
        self.sigma = sigma
        self.n_centers = int(n_centers)
        self.center_target = bool(center_target)  # classic KLIEP often centers on target
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.clip_min = float(clip_min)

        self.centers_ = None
        self.alpha_ = None
        self.sigma_ = None

    def get_params(self):
        return dict(sigma=self.sigma, n_centers=self.n_centers, center_target=self.center_target,
                    lr=self.lr, max_iter=self.max_iter, tol=self.tol,
                    random_state=self.random_state, clip_min=self.clip_min)

    def fit(self, Xs, Xt):
        Xs = np.asarray(Xs, float); Xt = np.asarray(Xt, float)
        ns, nt = Xs.shape[0], Xt.shape[0]
        rng = np.random.default_rng(self.random_state)
        # centers
        C = Xt[rng.choice(nt, size=min(self.n_centers, nt), replace=False)] if self.center_target \
            else Xs[rng.choice(ns, size=min(self.n_centers, ns), replace=False)]
        self.centers_ = C
        sigma = self.sigma if self.sigma is not None else _median_heuristic(Xs, Xt, seed=self.random_state)
        self.sigma_ = sigma

        K_sc = _pairwise_rbf(Xs, C, sigma)    # (ns, M)
        K_tc = _pairwise_rbf(Xt, C, sigma)    # (nt, M)

        # init alpha >= 0 and normalized so mean_s w(x)=1
        M = K_sc.shape[1]
        alpha = np.ones(M) / (np.mean(K_sc @ np.ones(M)) + 1e-12)

        last_obj = -np.inf
        for _ in range(self.max_iter):
            w_t = K_tc @ alpha
            w_t = np.maximum(w_t, self.clip_min)
            obj = np.mean(np.log(w_t))  # objective to maximize

            # gradient wrt alpha: (1/nt) sum_t K_tc(t, :) / w_t(t)
            grad = (K_tc / w_t[:, None]).mean(axis=0)

            # gradient ascent
            alpha = alpha + self.lr * grad
            alpha = np.maximum(alpha, 0.0)

            # enforce equality constraint: mean_s w_s = 1
            mean_ws = np.mean(K_sc @ alpha)
            if mean_ws <= 0:
                # reset small positive vector if collapsed
                alpha = np.ones(M) * 1e-6
                mean_ws = np.mean(K_sc @ alpha)
            alpha = alpha / (mean_ws + 1e-12)

            # convergence check
            if obj - last_obj < self.tol * (1.0 + abs(last_obj)):
                break
            last_obj = obj

        self.alpha_ = alpha
        return self

    def predict(self, X):
        if self.centers_ is None or self.alpha_ is None or self.sigma_ is None:
            raise RuntimeError("KLIEP estimator not fitted.")
        K = _pairwise_rbf(np.asarray(X, float), self.centers_, self.sigma_)
        w = K @ self.alpha_
        return np.maximum(w, self.clip_min)
