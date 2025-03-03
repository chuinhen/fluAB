import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


import statsmodels.api as sm
import scipy.stats as stats


file = r'E:/1 PROJECT/RESEARCH PROJECTS/HTJ res/influenza/Data/flu/data_flu_work.xlsx'
df = pd.read_excel(file)



# stats summary for continous variables (Pre-cleaning)
selected_columns_contVar = df.iloc[:, 78:95]

# Calculate percentiles, lower, upper bounds, 0.01th, and 99.98th percentiles
percentiles_ori = selected_columns_contVar.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.998]).transpose()

# Calculate lower and upper bounds
IQR = percentiles_ori['75%'] - percentiles_ori['25%']
lower_bound = percentiles_ori['25%'] - 1.5 * IQR
upper_bound = percentiles_ori['75%'] + 1.5 * IQR

# Add lower and upper bounds to the percentiles DataFrame
percentiles_ori['lower_bound'] = lower_bound
percentiles_ori['upper_bound'] = upper_bound

# Display the result
print("Stats Summary with Percentiles, Lower, Upper Bounds, 0.01th, and 99.8th Percentiles:")
print(percentiles_ori)


## DATA PREPROCESSING ##
# check for duplicates
duplicates = df[df.duplicated()]
print(duplicates)

# check for outliers
# Create individual horizontal box plots for each continuous variable
plt.figure(figsize=(15, 20))
for i, column in enumerate(selected_columns_contVar, 1):
    plt.subplot(6, 5, i)
    sns.boxplot(y=df[column], width=0.3, color='lightblue', linewidth=2)
    plt.title(f'Box Plot for {column}', fontsize=10)
    plt.ylabel(column, fontsize=11)

# Adjust layout to prevent overlapping
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the horizontal and vertical spacing
plt.show()


# Outlier detection using the IQR method:
outlier_threshold = 1.5
# Pre-calculate quantiles for the continuous variables
Q1 = selected_columns_contVar.quantile(0.25)
Q3 = selected_columns_contVar.quantile(0.75)
IQR_values = Q3 - Q1
lower_limits = Q1 - outlier_threshold * IQR_values
upper_limits = Q3 + outlier_threshold * IQR_values

# Compute outlier flags: values below lower_limits or above upper_limits
outlier_flags = (selected_columns_contVar < lower_limits) | (selected_columns_contVar > upper_limits)
outliers_percentage = outlier_flags.mean() * 100  # Mean gives the fraction of outliers per column

print("Outlier Percentage for Each Continuous Variable:")
print(outliers_percentage)


# check for missing vales 
# Replace empty strings with NaN in the selected continuous columns
df.iloc[:, 78:96] = df.iloc[:, 78:95].replace(r'^\s*$', np.nan, regex=True)

# Check for missing values in these continuous columns
missing_values = df.iloc[:, 78:95].isnull().sum()
percentage_missing = (missing_values / len(df)) * 100

print("\nPercentage Missing for Selected Continuous Columns (Columns 78 to 95):\n", percentage_missing)



import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

# Define the variables of interest
selected_vars = ['TWC', 'ALC', 'Platelet', 'CRP', 'bilirubin', 'Albumin', 'ALT', 'ALP']

# Subset the DataFrame to only include these variables
df_subset = df[selected_vars]

# --- Monkey-patch Axes.grid to fix the keyword error in missingno ---
old_grid = maxes.Axes.grid
def new_grid(self, *args, **kwargs):
    # Replace 'b' with 'visible' if present
    if 'b' in kwargs:
        kwargs['visible'] = kwargs.pop('b')
    return old_grid(self, *args, **kwargs)
maxes.Axes.grid = new_grid
# --- End of monkey-patch ---

# Visualize the missingness pattern for the selected variables
msno.matrix(df_subset)
plt.title("Missing Data Pattern for Selected Variables")
plt.show()

# Revert the monkey-patch so it doesn't affect other plots
maxes.Axes.grid = old_grid

# Calculate and print the missing value percentage for each selected variable
missing_summary = df_subset.isnull().sum() / len(df_subset) * 100
print("Missing Value Percentage for Selected Variables:\n", missing_summary)


import numpy as np
import pandas as pd
from scipy.stats import chi2

def little_mcar_test(data):
    """
    Perform Little's MCAR test on a pandas DataFrame.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the variables of interest.
        
    Returns:
        T_stat (float): The Little's test statistic.
        df_total (int): Degrees of freedom.
        p_value (float): The p-value for the test.
    """
    # Remove rows that are entirely missing
    data = data.dropna(how='all')
    
    # Calculate overall mean and covariance matrix from complete cases
    complete_data = data.dropna()
    mu = complete_data.mean()
    S = complete_data.cov()
    
    T_stat = 0
    df_total = 0
    
    # Identify unique missingness patterns (as binary mask rows)
    patterns = data.isnull().astype(int).drop_duplicates()
    
    # Loop over each unique missing data pattern
    for _, pattern in patterns.iterrows():
        # Determine which columns are observed in this pattern
        observed_cols = pattern[pattern == 0].index.tolist()
        if len(observed_cols) == 0:
            continue  # Skip if no variables are observed
        
        # Create a boolean mask for rows matching this pattern
        pattern_mask = (data[pattern.index].isnull().astype(int) == pattern).all(axis=1)
        group = data.loc[pattern_mask, observed_cols]
        n_g = group.shape[0]
        if n_g == 0:
            continue
        
        # Calculate the mean for this group (only on observed columns)
        group_mean = group.mean()
        mu_obs = mu[observed_cols]
        
        # Extract the sub-covariance matrix for the observed columns
        S_obs = S.loc[observed_cols, observed_cols]
        try:
            inv_S_obs = np.linalg.inv(S_obs.values)
        except np.linalg.LinAlgError:
            # If inversion fails, skip this group
            continue
        
        diff = (group_mean - mu_obs).values.reshape(-1, 1)
        # Contribution of this group to the test statistic
        T_g = n_g * np.dot(diff.T, np.dot(inv_S_obs, diff))
        T_stat += T_g[0, 0]
        # Degrees of freedom: number of observed variables in this pattern
        df_total += len(observed_cols)
    
    # Calculate the p-value using the chi-square distribution
    p_value = 1 - chi2.cdf(T_stat, df_total)
    return T_stat, df_total, p_value

# Define the variables to analyze
selected_vars = ['TWC', 'ALC', 'Platelet', 'CRP', 'bilirubin', 'Albumin', 'ALT', 'ALP']

# Subset the DataFrame to only these variables
df_mcar = df[selected_vars]

# Perform Little's MCAR test
T_stat, df_degrees, p_value = little_mcar_test(df_mcar)

print(f"Little's MCAR Test Statistic: {T_stat:.2f}")
print(f"Degrees of Freedom: {df_degrees}")
print(f"P-value: {p_value:.4f}")


# Remove variables with missingness 60%
columns_to_remove_missingness = ['Directbilirubin', 'CK']
df_delete = df.drop(columns=columns_to_remove_missingness)


# addressing Missing values 
# prepare dataset for MICE imputations
df_forMICE = df_delete.iloc[:, 3:93].copy()
# do not impute 
columns_to_drop = ['invasiveventilationdays', 'durationoxygenincludevent']  
df_forMICE.drop(columns=columns_to_drop, inplace=True)

rf_estimator = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,         # or set to a value like 10 or 15 to limit depth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=0
)

# Create the IterativeImputer instance using the configured RandomForestRegressor
imputer = IterativeImputer(
    estimator=rf_estimator,
    max_iter=10,
    random_state=0,
    sample_posterior=False  # Typically False for tree-based models
)
# Perform the imputation on your dataset
df_imputed = pd.DataFrame(imputer.fit_transform(df_forMICE), columns=df_forMICE.columns)



# Combine the imputed columns with the 'not to impute' columns
df_M = pd.concat([df_delete.iloc[:, :3], df_imputed, df_delete[columns_to_drop], df_delete.iloc[:, 93:]], axis=1)

# check missingness after MICE 
missing_values = df_M.isnull().sum()
percentage_missing_MICE = (missing_values / len(df_M)) * 100
print("\nPercentage Missing:\n", percentage_missing_MICE)

# stats summary for continous variables (after imputation)
selected_columns_imputed = df_M.iloc[:, 83:96]

# Calculate percentiles, lower, upper bounds, 0.01th, and 99.98th percentiles
percentiles_imputed = selected_columns_imputed.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.998]).transpose()

# Calculate lower and upper bounds
IQR = percentiles_imputed['75%'] - percentiles_imputed['25%']
lower_bound = percentiles_imputed['25%'] - 1.5 * IQR
upper_bound = percentiles_imputed['75%'] + 1.5 * IQR

# Add lower and upper bounds to the percentiles DataFrame
percentiles_imputed['lower_bound'] = lower_bound
percentiles_imputed['upper_bound'] = upper_bound

# Display the result
print("Stats Summary with Percentiles, Lower, Upper Bounds, 0.01th, and 99.8th Percentiles:")
print(percentiles_imputed)


# Define the list of variables for diagnostics
variables = ['TWC', 'ALC', 'Platelet', 'CRP', 'bilirubin', 'Albumin', 'ALT', 'ALP']

# Loop over each variable to generate individual plots
for var in variables:
    # Extract observed values from the original dataset (df_delete)
    observed = df_delete[var].dropna()
    
    # Extract imputed values from df_M at indices where the original dataset was missing
    imputed = df_M[var].dropna()
    
    plt.figure(figsize=(10, 5))
    sns.histplot(observed, color='blue', label='Observed', kde=True, stat="density", alpha=0.3)
    sns.histplot(imputed, color='red', label='Imputed', kde=True, stat="density", alpha=0.3)
    plt.title(f"Distribution Comparison for {var}")
    plt.legend()
    plt.show()
    
# Normality Assesment
# Check normality using Kolmogorov–Smirnov test and QQ plot

cont_variables_df_M = df_M.columns[77:93]

for column in cont_variables_df_M:
    # Exclude missing values before conducting tests
    data = df_M[column].dropna()

    # Kolmogorov–Smirnov test
    kstest_result = stats.kstest(data, 'norm')
    p_value_formatted = "{:.3f}".format(kstest_result.pvalue)
    print(f'Kolmogorov–Smirnov test for {column}: p-value = {p_value_formatted}')

# Combine QQ plots into subplots
num_plots = len(cont_variables_df_M)
num_cols = 6
num_rows = (num_plots + num_cols - 1) // num_cols  # Adjust to include the last row

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

for i, column in enumerate(cont_variables_df_M):
    row = i // num_cols
    col = i % num_cols

    # QQ plot
    sm.qqplot(df_M[column].dropna(), line='s', ax=axes[row, col])
    axes[row, col].set_title(f'QQ Plot for {column}')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# Save dataset
df_processed = df_M.iloc[:, 0:93].copy()
# csv_file_path = r'E:/1 PROJECT/RESEARCH PROJECTS/HTJ res/influenza/Data/flu/data_flu_processed.csv'
# df_processed.to_csv(csv_file_path, index=False)

# excel_file_path = r'E:/1 PROJECT/RESEARCH PROJECTS/HTJ res/influenza/Data/FluCovid_processedData.xlsx'
# df_processed.to_excel(excel_file_path, index=False)