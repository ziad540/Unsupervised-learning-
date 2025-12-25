import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing classes from your GMM.py file
from gmmBase import GaussianMixtureModel, StandardScaler

# 1. Load Data
file_path = '/Volumes/ELHOSS SSD/ML/labs/lab4/data.csv'
df = pd.read_csv(file_path)

# 2. Preprocess
X_cols = [c for c in df.columns if c not in ['id', 'diagnosis', 'Unnamed: 32']]
X = StandardScaler().fit_transform(df[X_cols].values)

# 3. Setup Experiment
ks = range(1, 8)
cov_types = ['full', 'tied', 'diag', 'spherical']
results = []
convergence_details = []

# 4. Run Analysis
print("Running GMM Experiments and Analyzing Convergence...")
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
        
        # Capture Convergence Analysis for k=2 (The most relevant cluster count)
        if k == 2:
            history = gmm.log_likelihood_history_
            for i, ll in enumerate(history):
                convergence_details.append({
                    'type': t,
                    'iteration': i + 1,
                    'avg_log_likelihood': ll
                })

# 5. Numeric Convergence Analysis Output
conv_df = pd.DataFrame(convergence_details)
res_df = pd.DataFrame(results)

print("\n--- NUMERIC CONVERGENCE ANALYSIS (K=2) ---")
# Pivot to show how each covariance type improves over iterations
convergence_pivot = conv_df.pivot(index='iteration', columns='type', values='avg_log_likelihood')
print(convergence_pivot.head(15)) # Show first 15 steps

# 6. Plotting Results

plt.figure(figsize=(14, 5))
for t in cov_types:
    sub = res_df[res_df['type'] == t]
    plt.subplot(1, 2, 1); plt.plot(sub['k'], sub['BIC'], label=t, marker='o'); plt.title('BIC Comparison')
    plt.subplot(1, 2, 2); plt.plot(sub['k'], sub['AIC'], label=t, marker='o'); plt.title('AIC Comparison')

plt.subplot(1, 2, 1); plt.xlabel('K'); plt.legend()
plt.subplot(1, 2, 2); plt.xlabel('K'); plt.legend()
plt.tight_layout()
plt.show()

# 7. Visualization of Convergence

plt.figure(figsize=(9, 6))
for t in cov_types:
    history = conv_df[conv_df['type'] == t]['avg_log_likelihood'].values
    plt.plot(history, label=f'Covariance: {t}', linewidth=2)

plt.title('Log-Likelihood Convergence Analysis (K=2)')
plt.xlabel('Iteration Number')
plt.ylabel('Average Log-Likelihood')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 8. Convergence Speed Summary
print("\n--- CONVERGENCE SPEED SUMMARY ---")
for t in cov_types:
    iters = conv_df[conv_df['type'] == t]['iteration'].max()
    final_ll = conv_df[conv_df['type'] == t]['avg_log_likelihood'].iloc[-1]
    print(f"Type: {t:10} | Iterations to Converge: {iters:3} | Final LL: {final_ll:.4f}")