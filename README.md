# CGDRO

CGDRO is a python library providing comprehensive multi-source prediction and model statistical inference without knowing target labels, offering multi-source solutions to low-dimensional and high-dimensional, linear and complex data, regression and classification problems.

CGDRO supports 4 modules:

| Python Module | Description | Statistical Inference |
|-----------------------------|---------------------------------|----------------|
| `Regression.linear.ld` | Linear prediction model (low-dimensional) | ✅ |
| `Regression.linear.hd` | High-dimensional linear model | ✅ |
| `Regression.ml` | Machine learning prediction model | ❌ |
| `Classification` | Linear model for classification task | ✅ |

# Installation

The development version from [GitHub](https://github.com) with:

``` python
pip install git+https://github.com/jaon11/CGDRO-Py.git
```

# Example

This is a basic example which shows you how to solve a common problem:

``` python
from cgdro.Regression import linear
```

The data is heterogeneous and covariates shift between source and target
data

``` python
# number of source groups = 3, with 1000 samples each
# sigma: source group 1,3: 0.5; source group 2: 2
# target sample size = 10000
# dimension p = 5
from cgdro import data
n_list = [1000, 1000, 1000]
N = 10000  # target sample size
data = data.DataContainerSimu_linear_reg_lowd(n_list=n_list, N=N, p=5)
data.generate_funcs_list(seed=0)
data.generate_data(seed=0)

Xlist = data.X_sources_list
Ylist = data.Y_sources_list
X0 = data.X_target

```

Fitting CGDRO model for low-dimensional linear regression with reward
loss and summarize results

``` python
## First announcing the module
## Then calling the functions fit() and infer()
## Note: only when loss_type='reward', infer() can be called to get confidence intervals
## For other loss_type, only point estimation and prediction can be done
reg = linear.ld()
reg.fit(Xlist, Ylist, X0, loss_type='reward')
reg.infer(alpha=0.05)
## summary
reg.summary()
```

    ##Model Summary:
    ##=================================
    ##CGDRO Aggregated Weights:

    ##group     |        1        2        3
    ##weight_   |   0.4567   0.3451   0.1982

    ##=================================
    ##Coefficient Estimators:

    ##index     |        1        2        3        4        5
    ##coef_     |  -0.0655  -0.0433   0.0032  -0.0018   0.0997

    ##=================================
    ##Confidence Intervals:

    ##index     |              1              2              3              4              5
    ##CI        | (-0.1283,-0.0027) (-0.1080,0.0214) (-0.0601,0.0665) (-0.0666,0.0629) (0.0351,0.1643)

Make prediction on target data

``` python
## predict
pred = reg.predict()
print(pred[:10])
```

    ## [-0.13644197 -0.08692431  0.04184841 -0.18107761 -0.30044671  0.0026928 -0.30307077 -0.18239394 -0.06712825  0.20524958]



# References

<span id="ref-guo2024statistical"></span>
**Guo, Z.** (2024). *Statistical inference for maximin effects: Identifying stable associations across multiple studies.*  
*Journal of the American Statistical Association*, 119(547), 1968–1984.  
[Paper link](https://doi.org/10.1080/01621459.2023.2258469)

<span id="ref-wang2023distributionally"></span>
**Wang, Z.**, **Bühlmann, P.**, & **Guo, Z.** (2023). *Distributionally robust machine learning with multi-source data.*  
*arXiv preprint* [arXiv:2309.02211](https://arxiv.org/abs/2309.02211)

<span id="ref-guo2025statistical"></span>
**Guo, Z.**, **Wang, Z.**, **Hu, Y.**, & **Bach, F.** (2025). *Statistical Inference for Conditional Group Distributionally Robust Optimization with Cross-Entropy Loss.*  
*arXiv preprint* [arXiv:2507.09905](https://arxiv.org/abs/2507.09905)
