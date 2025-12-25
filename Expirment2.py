import pandas as pd
import matplotlib.pyplot as plt
from gmmBase import GaussianMixtureModel, StandardScaler

# 1. Load Data
# Update the path if necessary
file_path = '/Volumes/ELHOSS SSD/ML/labs/lab4/data.csv'
df = pd.read_csv(file_path)

# 2. Preprocess
X_cols = [c for c in df.columns if c not in ['id', 'diagnosis', 'Unnamed: 32']]
X = StandardScaler().fit_transform(df[X_cols].values)

# 3. Setup Experiment
ks = range(1, 8)
cov_types = ['full', 'tied', 'diag', 'spherical']
results = []
convergence_data = {}

# 4. Run Analysis
print("Running GMM Experiments...")
for t in cov_types:
    for k in ks:
        # Create and fit model
        gmm = GaussianMixtureModel(n_components=k, covariance_type=t, random_state=42).fit(X)
        
        # Save AIC/BIC for model selection
        results.append({
            'k': k, 
            'type': t, 
            'BIC': gmm.bic(X), 
            'AIC': gmm.aic(X)
        })
        
        # Save convergence for k=2 specifically
        if k == 2:
            convergence_data[t] = gmm.log_likelihood_history_

res_df = pd.DataFrame(results)

# 5. Output Numeric Results
print("\n--- AIC/BIC SUMMARY ---")
print(res_df.sort_values(by='BIC').head(10).to_string(index=False))

# 6. Plotting

plt.figure(figsize=(14, 5))
for t in cov_types:
    sub = res_df[res_df['type'] == t]
    plt.subplot(1, 2, 1); plt.plot(sub['k'], sub['BIC'], label=t, marker='o'); plt.title('BIC Comparison')
    plt.subplot(1, 2, 2); plt.plot(sub['k'], sub['AIC'], label=t, marker='o'); plt.title('AIC Comparison')

plt.subplot(1, 2, 1); plt.xlabel('Number of Clusters (K)'); plt.legend()
plt.subplot(1, 2, 2); plt.xlabel('Number of Clusters (K)'); plt.legend()
plt.tight_layout()
plt.show()

# Convergence Plot

plt.figure(figsize=(8, 5))
for t, history in convergence_data.items():
    plt.plot(history, label=f'{t} cov')
plt.title('Log-Likelihood Convergence History (K=2)')
plt.xlabel('Iteration'); plt.ylabel('Avg Log-Likelihood'); plt.legend(); plt.grid(True)
plt.show()