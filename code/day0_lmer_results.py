#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:18:53 2025
Updated on Aug 15 2025 by ChatGPT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# === Paths and variable names ===
input_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day0_filt_noKO.csv'
output_path = '/Users/bass/Downloads/LaurenOh_FC/results/day0_lmer_results_anova.xlsx'
plots_dir = '/Users/bass/Downloads/LaurenOh_FC/results/plots/'
dependent_var = 'Pct Total Time Freezing'

# === Load and clean data ===
df = pd.read_csv(input_path)
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# Only keep rows where Component Name starts with "Sound" (not "Shock")
df = df[df['Component Name'].str.startswith('Sound')].copy()

# Convert relevant columns to categorical
categorical_vars = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name']
for col in categorical_vars:
    if col in df.columns:
        df[col] = df[col].astype('category')

# === Fit mixed model ===
formula = f'Q("{dependent_var}") ~ APOE + Diet + Lifestyle + HN + Sex + Age_Months + Q("Component Name")'
model = mixedlm(formula, df, groups=df["Animal ID"]).fit()

# === Summary: coefficients, p-values, FDR ===
summary_df = pd.DataFrame({
    'Effect': model.params.index,
    'Coefficient': model.params.values,
    'Std_Err': model.bse,
    'p-value': model.pvalues
})
summary_df = summary_df[summary_df['Effect'] != 'Intercept']

# Adjust p-values (FDR)
valid_pvals = summary_df['p-value'].dropna()
_, pvals_fdr, _, _ = multipletests(valid_pvals, method='fdr_bh')
summary_df.loc[valid_pvals.index, 'FDR p-value'] = pvals_fdr

# Compute Cohen's d
summary_df['Cohens_d'] = summary_df['Coefficient'] / summary_df['Std_Err']

# === Tukey posthoc tests ===
factors = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name']
posthoc_dfs = []
for factor in factors:
    try:
        tukey = pairwise_tukeyhsd(endog=df[dependent_var], groups=df[factor], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df['Factor'] = factor
        posthoc_dfs.append(tukey_df)
    except Exception as e:
        print(f"Could not run Tukey for {factor}: {e}")

# Combine posthoc results
all_posthoc_combined = pd.concat(posthoc_dfs, ignore_index=True)

# === Save results to Excel ===
os.makedirs(plots_dir, exist_ok=True)
with pd.ExcelWriter(output_path, mode='w') as writer:
    summary_df.to_excel(writer, sheet_name='MixedLM_Effects', index=False)
    all_posthoc_combined.to_excel(writer, sheet_name='Tukey_Posthoc', index=False)

# === Save bar plots ===
for factor in factors:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=factor, y=dependent_var, data=df, ci='sd', palette='Set2')
    plt.title(f'{dependent_var} by {factor}')
    plt.ylabel(dependent_var)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{factor}_barplot.png'))
    plt.close()

print(f"✅ Results saved to:\n{output_path}\nPlots saved to: {plots_dir}")
