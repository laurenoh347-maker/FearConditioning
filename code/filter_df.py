#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:06:53 2025

@author: bass
"""
#filter abd edit data frame 
import pandas as pd

# Step 1: Read the input CSV
input_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day0_combined_metadata.csv'
df = pd.read_csv(input_path)

# Step 2: Replace 'APOE_KO' with 'KO' in the 'Genotype' column
df['Genotype'] = df['Genotype'].replace('APOE_KO', 'KO')

# Step 3: Filter to keep only specific Genotype values
valid_genotypes = ['APOE22HN', 'APOE33HN', 'APOE44HN', 'APOE22', 'APOE33', 'APOE44']
df_filtered = df[df['Genotype'].isin(valid_genotypes)].copy()

# Step 4: Define functions to extract APOE and HN
def extract_apoe(genotype):
    if '22' in genotype:
        return 'APOE2'
    elif '33' in genotype:
        return 'APOE3'
    elif '44' in genotype:
        return 'APOE4'
    else:
        return None

def extract_hn(genotype):
    return 1 if 'HN' in genotype else 0

# Step 5: Apply functions to create new columns
df_filtered['APOE'] = df_filtered['Genotype'].apply(extract_apoe)
df_filtered['HN'] = df_filtered['Genotype'].apply(extract_hn)

# Step 6: Save the filtered DataFrame to a new CSV
output_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day0_filt_noKO.csv'
df_filtered.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")