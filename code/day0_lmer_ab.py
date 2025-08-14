#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:18:53 2025
@author: bass
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
output_path = '/Users/bass/Downloads/LaurenOh_FC/results/day0_anova.xlsx'
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
valid_pvals = summary_df['p-value'].dropna()
_, pvals_fdr, _, _ = multipletests(valid_pvals, method='fdr_bh')
summary_df.loc[valid_pvals.index, 'FDR p-value'] = pvals_fdr
summary_df['Abs_Coeff'] = summary_df['Coefficient'].abs()

# === Group means (example by Diet) ===
group_means = df.groupby('Diet')[dependent_var].mean().reset_index()
group_means.columns = ['Diet', 'Group Mean']

# === Tukey posthoc tests ===
factors = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name']
posthoc_results = {}
for factor in factors:
    try:
        tukey = pairwise_tukeyhsd(endog=df[dependent_var], groups=df[factor], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        posthoc_results[factor] = tukey_df
    except Exception as e:
        print(f"Could not run Tukey for {factor}: {e}")

# === Group counts and age range ===
group_counts_main = df.groupby(['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name']).size().reset_index(name='Count')
age_range_df = pd.DataFrame({
    'Age_Months_Min': [df['Age_Months'].min()],
    'Age_Months_Max': [df['Age_Months'].max()]
})

# === Combine all posthoc results to one sheet ===
posthoc_dfs = []
for factor, tukey_df in posthoc_results.items():
    tukey_df['Factor'] = factor
    sep_row = pd.DataFrame([[''] * len(tukey_df.columns)], columns=tukey_df.columns)
    posthoc_dfs.append(sep_row)
    posthoc_dfs.append(tukey_df)
all_posthoc_combined = pd.concat(posthoc_dfs, ignore_index=True)

# === Save results to Excel ===
with pd.ExcelWriter(output_path, mode='w') as writer:
    summary_df.to_excel(writer, sheet_name='MixedLM_Effects', index=False)
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    group_counts_main.to_excel(writer, sheet_name='Group Counts', index=False)
    age_range_df.to_excel(writer, sheet_name='Age_Months_Range', index=False)
    all_posthoc_combined.to_excel(writer, sheet_name='Tukey_All_Posthoc', index=False)
    for factor, tukey_df in posthoc_results.items():
        tukey_df.to_excel(writer, sheet_name=f'Tukey_{factor}', index=False)

# === Save bar plots ===
os.makedirs(plots_dir, exist_ok=True)
for factor in factors:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=factor, y=dependent_var, data=df, ci='sd', palette='Set2')
    plt.title(f'{dependent_var} by {factor}')
    plt.ylabel(dependent_var)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{factor}_barplot.png'))
    plt.close()

print(f"✅ Mixed model results, posthoc tests, and plots saved to:\n{output_path}\n{plots_dir}")
