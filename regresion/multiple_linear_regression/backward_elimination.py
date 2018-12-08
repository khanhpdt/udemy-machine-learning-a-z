import statsmodels.api as sm
import numpy as np


def backward_elimination(features_input, labels, significance_level=0.05, add_intercept=True):
    n_examples = features_input.shape[0]

    if add_intercept:
        features = np.append(arr=np.ones((n_examples, 1)).astype(int), values=features_input, axis=1).astype(float)
    else:
        features = features_input.copy()

    n_features = features.shape[1]
    features_opt_idxs = list(range(n_features))

    features_opt_found = False
    while not features_opt_found and len(features_opt_idxs) > 0:
        regressor_ols = sm.OLS(endog=labels, exog=features[:, features_opt_idxs]).fit()

        pvalues = regressor_ols.pvalues
        max_pvalue_idx = pvalues.argmax()

        if pvalues[max_pvalue_idx] > significance_level:
            features_opt_idxs.pop(max_pvalue_idx)
        else:
            features_opt_found = True

    # remove the feature corresponding to the intercept term and shift the other features
    # so that their indexes will match the input features
    if add_intercept:
        features_opt_idxs = [i - 1 for i in features_opt_idxs if i > 0]

    return features_opt_idxs
