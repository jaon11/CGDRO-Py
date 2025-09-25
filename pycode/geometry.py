import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def _project_to_simplex(v):
    """
    Project vector v onto the probability simplex {w: w>=0, sum w = 1}.
    """
    v = np.asarray(v, float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u > cssv / (np.arange(n) + 1))[0][-1]
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

# -----------------------------
# (1a) Nearest point to origin in the AFFINE HULL of points
# -----------------------------
def nearest_on_affine_hull(P):
    """
    P: (L, p) points as rows.
    Returns x_aff (p,), the point in affine hull(P) with minimal ||x||,
    and the barycentric weights w (L,) s.t. x_aff = P^T w, sum(w)=1.
    """
    P = np.asarray(P, float)        # L x p
    L, p = P.shape
    G = P @ P.T                     # L x L
    e = np.ones(L)
    # Solve KKT system: [ G  e ][w] = [0]
    #                    [ e^T 0 ][λ]   [1]
    KKT = np.block([[G, e[:, None]],
                    [e[None, :], np.zeros((1, 1))]])
    rhs = np.concatenate([np.zeros(L), np.array([1.0])])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    w = sol[:L]
    x_aff = (w @ P)                 # p,
    return x_aff, w

# -----------------------------
# (1b) Nearest point to origin in the CONVEX HULL of points
#      (quadratic program via projected gradient on simplex)
# -----------------------------
def nearest_on_convex_hull(P, max_iter=10_000, tol=1e-10, step=None):
    """
    Minimize ||P^T w||^2 subject to w in simplex (w>=0, sum w=1).
    Returns x_ch (p,), optimal weights w (L,).
    Uses simple projected gradient with exact simplex projection.
    """
    P = np.asarray(P, float)        # L x p
    L, p = P.shape
    G = P @ P.T                     # L x L, objective = w^T G w
    # Lipschitz constant for gradient = 2 * ||G||_2; estimate via power iteration
    if step is None:
        y = np.random.randn(L)
        for _ in range(50):
            y = G @ y
            y /= np.linalg.norm(y) + 1e-15
        Lg = 2.0 * (y @ (G @ y))
        step = 1.0 / (Lg + 1e-15)

    w = np.ones(L) / L
    prev = np.inf
    for _ in range(max_iter):
        grad = 2.0 * (G @ w)
        w = _project_to_simplex(w - step * grad)
        val = w @ (G @ w)
        if abs(prev - val) < tol * max(1.0, prev):
            break
        prev = val
    x_ch = w @ P
    return x_ch, w

import numpy as np

def circumcenter_3vectors(P, clip_tol=1e-12):
    """
    Compute the circumcenter of 3 vectors (points) in R^p.

    Parameters
    ----------
    P : array-like, shape (3, p)
        Rows are the three input vectors x1, x2, x3.
    clip_tol : float
        Numerical tolerance for clipping very small coefficients to 0.

    Returns
    -------
    c : ndarray, shape (p,)
        The circumcenter (center of the circle passing through x1, x2, x3).
    r : float
        The radius of the circumcircle.
    alpha : ndarray, shape (3,)
        The weights such that c = alpha1*x1 + alpha2*x2 + alpha3*x3,
        with sum(alpha) = 1. These are the barycentric coordinates of
        the circumcenter with respect to the three input points.

    Notes
    -----
    - The three input points must not be collinear (otherwise the
      circumcircle is undefined).
    - Works for any dimension p >= 2 (the three points define a 2D plane
      where the circumcircle lives).
    """
    P = np.asarray(P, float)
    assert P.shape[0] == 3, "P must contain exactly 3 vectors (rows)."

    # Compute Gram matrix and squared norms of the three points
    B = P @ P.T        # 3x3 Gram matrix
    d = np.diag(B)     # squared norms of x1, x2, x3

    # Build linear system A alpha = b
    # Equal-distance constraints: ||c - x1||^2 = ||c - x2||^2 = ||c - x3||^2
    A = np.zeros((3, 3))
    b = np.zeros(3)

    # Equation for (x2 vs x1)
    A[0, :] = 2.0 * (B[1, :] - B[0, :])
    b[0]    =        d[1]   - d[0]
    # Equation for (x3 vs x1)
    A[1, :] = 2.0 * (B[2, :] - B[0, :])
    b[1]    =        d[2]   - d[0]
    # Constraint: sum(alpha) = 1
    A[2, :] = 1.0
    b[2]    = 1.0

    # Solve for weights alpha
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("Points are collinear or nearly collinear; circumcircle undefined.")
    alpha = np.linalg.solve(A, b)

    # Numerical cleanup: clip tiny values and renormalize
    alpha[np.abs(alpha) < clip_tol] = 0.0
    s = alpha.sum()
    if s != 0:
        alpha = alpha / s

    # Circumcenter and radius
    c = alpha @ P
    r = np.linalg.norm(P[0] - c)

    return c, r, alpha
