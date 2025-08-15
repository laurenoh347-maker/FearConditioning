#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:20:00 2025
@author: bass
Day1 ANOVA + posthoc + combined Excel output
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# === Paths and variable names ===
input_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day1_filt_noKO.csv'
output_path = '/Users/bass/Downloads/LaurenOh_FC/results/day1_anova2.xlsx'
dependent_var = 'Pct Total Time Freezing'

# === Load data ===
df = pd.read_csv(input_path)

# Strip whitespace from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# === Ensure categorical variables are treated as categories ===
categorical_vars = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']
for col in categorical_vars:
    if col in df.columns:
        df[col] = df[col].astype('category')

# === Build and fit linear model ===
formula = f'Q("{dependent_var}") ~ APOE + Diet + Lifestyle + HN + Sex + Age_Months + Diet:APOE + Diet:HN + Diet:Lifestyle'
model = smf.ols(formula, data=df).fit()

# === ANOVA table ===
anova_results = anova_lm(model, typ=2)

# === Apply FDR correction ===
fdr_col = [None] * len(anova_results)  # placeholder
valid_pvals = anova_results['PR(>F)'].dropna()
_, pvals_fdr, _, _ = multipletests(valid_pvals, method='fdr_bh')

# Assign FDR only to valid rows
for idx, fdr_val in zip(valid_pvals.index, pvals_fdr):
    fdr_col[anova_results.index.get_loc(idx)] = fdr_val

anova_results['FDR p-value'] = fdr_col

# === Group means for Diet ===
group_means = df.groupby('Diet')[dependent_var].mean().reset_index()
group_means.columns = ['Diet', 'Group Mean']

# === Tukey HSD for each factor ===
factors = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']
posthoc_results = {}

for factor in factors:
    try:
        tukey = pairwise_tukeyhsd(endog=df[dependent_var],
                                  groups=df[factor],
                                  alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        posthoc_results[factor] = tukey_df
    except Exception as e:
        print(f"Could not run Tukey for {factor}: {e}")

# === Counts per group ===
group_counts_main = df.groupby(['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']).size().reset_index(name='Count')

# === Age range ===
age_range_df = pd.DataFrame({
    'Age_Months_Min': [df['Age_Months'].min()],
    'Age_Months_Max': [df['Age_Months'].max()]
})

# === Combine all posthoc results into one sheet ===
posthoc_dfs = []
for factor, tukey_df in posthoc_results.items():
    tukey_df['Factor'] = factor
    posthoc_dfs.append(tukey_df)
    posthoc_dfs.append(pd.DataFrame([[''] * len(tukey_df.columns)], columns=tukey_df.columns))  # separator

all_posthoc_combined = pd.concat(posthoc_dfs, ignore_index=True)

# === Save to Excel ===
with pd.ExcelWriter(output_path) as writer:
    anova_results.to_excel(writer, sheet_name='ANOVA')
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    for factor, tukey_df in posthoc_results.items():
        tukey_df.to_excel(writer, sheet_name=f'Tukey_{factor}', index=False)
    group_counts_main.to_excel(writer, sheet_name='Group Counts', index=False)
    age_range_df.to_excel(writer, sheet_name='Age_Months_Range', index=False)
    all_posthoc_combined.to_excel(writer, sheet_name='Tukey_All_Posthoc', index=False)

print(f"✅ Day1 ANOVA + posthoc results saved to: {output_path}")
