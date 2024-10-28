

# Fit the OLS model using statsmodels
model = sm.OLS(y, X)
results = model.fit()

# Same as:

X_transpose = X.T
X_transpose_X = X_transpose @ X
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_y = X_transpose @ y
betas_hat1 = X_transpose_X_inv @ X_transpose_y
# Get the coefficients
betas_hat = results.params

residuals = y - X @ betas_hat

# Calculate the mean squared error
MSE = np.sum(residuals**2) / (len(y) - X.shape[1]) # unbiased estimae of sigma

# Calculate the variance-covariance matrix of the betas
var_betas = MSE * X_transpose_X_inv.diagonal()

# Calculate standard errors of the betas
std_err_betas = np.sqrt(var_betas)

# Calculate t-statistics for each beta
t_stats = betas_hat / std_err_betas

# run t test 
t_test = results.t_test(np.eye(len(betas_hat)))

# run t test manually
t_test_manual = (betas_hat / std_err_betas)
t_test_manual = pd.DataFrame(t_test_manual, columns=['t_stat'])
t_test_manual['p_value'] = 2 * (1 - stats.t.cdf(np.abs(t_test_manual['t_stat']), len(y) - X.shape[1]))
t_test_manual['reject'] = t_test_manual['p_value'] < 0.05
t_test_manual.index = feature_names
t_test_manual
