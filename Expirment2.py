import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GMM import GaussianMixtureModel, StandardScaler

# 1. LOAD AND PREPARE DATA
file_path = '/Volumes/ELHOSS SSD/ML/labs/lab4/data.csv'
df = pd.read_csv(file_path)
X_cols = [c for c in df.columns if c not in ['id', 'diagnosis', 'Unnamed: 32']]
X = StandardScaler().fit_transform(df[X_cols].values)

# 2. CONFIGURATION
ks = range(1, 8)
cov_types = ['full', 'tied', 'diag', 'spherical']
results = []
convergence_log = []

# 3. EXECUTE EXPERIMENTS
print("Analyzing GMM configurations for AIC, BIC, and Convergence...")

for t in cov_types:
    for k in ks:
        model = GaussianMixtureModel(n_components=k, covariance_type=t, random_state=42).fit(X)
        
        # Numeric Analysis: BIC and AIC
        results.append({
            'K': k, 
            'Covariance_Type': t, 
            'BIC': round(model.bic(X), 2), 
            'AIC': round(model.aic(X), 2)
        })
        
        # Log-Likelihood Convergence for K=2 (Standard comparison)
        if k == 2:
            for i, ll in enumerate(model.log_likelihood_history_):
                convergence_log.append({
                    'Iteration': i + 1, 'Type': t, 'Avg_LL': ll
                })

# 4. NUMERIC TABLES OUTPUT
res_df = pd.DataFrame(results)
conv_df = pd.DataFrame(convergence_log)

print("\n" + "="*60)
print("1. OPTIMAL COMPONENTS ANALYSIS (AIC & BIC)")
print("="*60)
print(res_df.sort_values(by='BIC').head(10).to_string(index=False))

print("\n" + "="*60)
print("2. LOG-LIKELIHOOD CONVERGENCE NUMERIC (K=2)")
print("="*60)
print(conv_df.pivot(index='Iteration', columns='Type', values='Avg_LL'))

# 5. FIRST PLOT: AIC & BIC SELECTION

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for t in cov_types:
    sub = res_df[res_df['Covariance_Type'] == t]
    plt.plot(sub['K'], sub['AIC'], label=t, marker='s')
plt.title('AIC Scores (Lower is Better)')
plt.xlabel('K'); plt.ylabel('AIC'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for t in cov_types:
    sub = res_df[res_df['Covariance_Type'] == t]
    plt.plot(sub['K'], sub['BIC'], label=t, marker='o')
plt.title('BIC Scores (Lower is Better)')
plt.xlabel('K'); plt.ylabel('BIC'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. SECOND PLOT: LOG-LIKELIHOOD CONVERGENCE

plt.figure(figsize=(10, 6))
for t in cov_types:
    history = conv_df[conv_df['Type'] == t]['Avg_LL'].values
    plt.plot(history, label=f'Covariance: {t}', linewidth=2)

plt.title('Log-Likelihood Convergence Analysis (at K=2)')
plt.xlabel('Iteration Number')
plt.ylabel('Average Log-Likelihood')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()