#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:18:53 2025

@author: bass
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# === Paths and variable names ===
input_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day0_filt_noKO.csv'
output_path = '/Users/bass/Downloads/LaurenOh_FC/results/day0_anova.xlsx'
dependent_var = 'Pct Total Time Freezing'

# === Load data ===
df = pd.read_csv(input_path)

# Strip whitespace from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
df = df[df['Component Name'].str.startswith('Sound')]

# === Ensure categorical variables are treated as categories ===
categorical_vars = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']
for col in categorical_vars:
    if col in df.columns:
        df[col] = df[col].astype('category')

# === Build and fit linear model ===
formula = f'Q("{dependent_var}") ~ APOE*HN*Age_Months*Sex*Diet*Lifestyle'
formula = 'Q("Pct Total Time Freezing") ~ APOE + Diet + Lifestyle + HN + Sex + Age_Months + Diet:APOE + Diet:HN + Diet:Lifestyle'

model = smf.ols(formula, data=df).fit()

# === ANOVA table ===
anova_results = anova_lm(model, typ=2)

# Extract only the rows with valid p-values (exclude Residual)
pvals = anova_results.loc[anova_results['PR(>F)'].notnull(), 'PR(>F)']

# Apply FDR correction
_, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

# Assign back to the correct rows
anova_results.loc[anova_results['PR(>F)'].notnull(), 'FDR p-value'] = pvals_fdr


# === Group means for Diet ===
group_means = df.groupby('Diet')[dependent_var].mean().reset_index()
group_means.columns = ['Diet', 'Group Mean']

# === Tukey HSD post-hoc test by Diet ===
tukey_result = pairwise_tukeyhsd(df[dependent_var], df['Diet'])
tukey_df = pd.DataFrame(tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])

# === Save results to Excel ===
with pd.ExcelWriter(output_path) as writer:
    anova_results.to_excel(writer, sheet_name='ANOVA')
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    tukey_df.to_excel(writer, sheet_name='Tukey Posthoc (Diet)', index=False)

print(f"✅ ANOVA and posthoc results saved to: {output_path}")


# Count number of samples per group
group_counts = df.groupby(['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']).size().reset_index(name='Count')

# Optional: print to check
print(group_counts)

# Save to Excel (add to your existing ExcelWriter block)
with pd.ExcelWriter(output_path) as writer:
    # Existing sheets
    anova_results.to_excel(writer, sheet_name='ANOVA')
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    tukey_df.to_excel(writer, sheet_name='Tukey Posthoc (Diet)', index=False)
    
    # ➕ Add this line
    group_counts.to_excel(writer, sheet_name='Group Counts', index=False)


import pandas as pd

# Count number of samples per combination of main factors
group_counts_main = df.groupby(['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']).size().reset_index(name='Count')

# Calculate Age_Months range
age_min = df['Age_Months'].min()
age_max = df['Age_Months'].max()
age_range_df = pd.DataFrame({'Age_Months_Min': [age_min], 'Age_Months_Max': [age_max]})


with pd.ExcelWriter(output_path) as writer:
    # Save ANOVA, means, posthoc, etc.
    anova_results.to_excel(writer, sheet_name='ANOVA')
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    tukey_df.to_excel(writer, sheet_name='Tukey Posthoc (Diet)', index=False)

    # ➕ New: Save counts and Age_Months range
    group_counts_main.to_excel(writer, sheet_name='Group Counts', index=False)
    age_range_df.to_excel(writer, sheet_name='Age_Months_Range', index=False)


from statsmodels.stats.multicomp import pairwise_tukeyhsd

factors = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle']

posthoc_results = {}

for factor in factors:
    try:
        tukey = pairwise_tukeyhsd(endog=df['Pct Total Time Freezing'],
                                  groups=df[factor],
                                  alpha=0.05)
        # Convert to DataFrame for easier saving/processing
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        posthoc_results[factor] = tukey_df
    except Exception as e:
        print(f"Could not run Tukey for {factor}: {e}")


with pd.ExcelWriter(output_path) as writer:
    anova_results.to_excel(writer, sheet_name='ANOVA')
    group_means.to_excel(writer, sheet_name='Group Means', index=False)
    
    # Save each factor's posthoc results
    for factor, tukey_df in posthoc_results.items():
        sheet_name = f'Tukey_{factor}'
        tukey_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    group_counts_main.to_excel(writer, sheet_name='Group Counts', index=False)
    age_range_df.to_excel(writer, sheet_name='Age_Months_Range', index=False)


import pandas as pd

posthoc_dfs = []

for factor, tukey_df in posthoc_results.items():
    # Add a column to label the factor
    tukey_df['Factor'] = factor
    
    # Optionally, add a separator row for readability
    sep_row = pd.DataFrame([[''] * len(tukey_df.columns)], columns=tukey_df.columns)
    
    # Append separator and results
    posthoc_dfs.append(sep_row)
    posthoc_dfs.append(tukey_df)

# Concatenate all
all_posthoc_combined = pd.concat(posthoc_dfs, ignore_index=True)

# Save to Excel tab
with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='replace') as writer:
    all_posthoc_combined.to_excel(writer, sheet_name='Tukey_All_Posthoc', index=False)

