#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMC-free Bayesian approximation via Bayesian (Dirichlet) bootstrap
+ Publication-ready posterior plots for APOE, with effect sizes.

- Respects clustering by Animal ID (weights first sampled at the cluster level)
- Computes posterior over group means AND within-group variances
- Tukey-style pairwise comparisons
- APOE: plots group posteriors & differences vs. APOE3 (baseline)
- Adds Cohen's d (per-draw) with 95% HDIs to comparisons
"""

import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

# ========= User config =========
input_path = '/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day0_filt_noKO.csv'
output_path_freq = '/Users/bass/Downloads/LaurenOh_FC/results/day0_anova.xlsx'
dependent_var = 'Pct Total Time Freezing'
factors = ['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name']
B = 4000
seed = 42
use_sound_filter = True
apoe_baseline = 'APOE3'
# ==============================

rng = np.random.default_rng(seed)

def hdi(samples, cred=0.95):
    """Highest density interval via sliding window on sorted samples."""
    x = np.sort(np.asarray(samples))
    n = x.size
    if n == 0:
        return np.nan, np.nan
    m = max(1, int(np.floor(cred * n)))
    widths = x[m:] - x[:n - m]
    j = np.argmin(widths)
    return x[j], x[j + m]

def _weighted_mean_var(y, w):
    """Weighted mean and (biased) variance using weights w (sum can be != 1)."""
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan
    m = np.sum(w * y) / sw
    v = np.sum(w * (y - m) ** 2) / sw
    return m, v

def bayes_bootstrap_group_means_cluster(df, value_col, group_col, cluster_col, B, rng):
    """
    Dirichlet bootstrap of group means and variances with cluster-respecting weights.
    Returns:
        levels (list[str]),
        mean_draws: dict[level] -> (B,) array,
        var_draws:  dict[level] -> (B,) array
    """
    clusters = df[cluster_col].astype('category')
    cluster_codes = clusters.cat.codes.values
    n_clusters = clusters.cat.categories.size
    cluster_rows = {c: np.where(cluster_codes == c)[0] for c in range(n_clusters)}

    groups = df[group_col].astype('category')
    levels = list(groups.cat.categories)
    group_idx = groups.cat.codes.values
    y = df[value_col].values

    level_masks = [group_idx == i for i in range(len(levels))]
    cluster_row_counts = {c: len(cluster_rows[c]) for c in range(n_clusters)}

    mean_draws = {lvl: np.empty(B, dtype=float) for lvl in levels}
    var_draws  = {lvl: np.empty(B, dtype=float) for lvl in levels}

    for b in range(B):
        # sample cluster weights, spread equally to rows within cluster
        w_cluster = rng.dirichlet(alpha=np.ones(n_clusters))
        w = np.empty(len(df), dtype=float)
        for c in range(n_clusters):
            rows_c = cluster_rows[c]
            w[rows_c] = w_cluster[c] / cluster_row_counts[c] if cluster_row_counts[c] > 0 else 0.0

        # group-level weighted mean & variance
        for i, lvl in enumerate(levels):
            mask = level_masks[i]
            m, v = _weighted_mean_var(y[mask], w[mask])
            mean_draws[lvl][b] = m
            var_draws[lvl][b]  = v

    return levels, mean_draws, var_draws

def summarize_posterior_means(levels, mean_draws):
    rows = []
    for lvl in levels:
        s = mean_draws[lvl]
        mean = np.nanmean(s)
        lo, hi = hdi(s[~np.isnan(s)], 0.95)
        rows.append({"Level": lvl, "Posterior Mean": mean, "HDI 2.5%": lo, "HDI 97.5%": hi})
    return pd.DataFrame(rows)

def pairwise_comparisons(levels, mean_draws, var_draws):
    """
    Pairwise diffs for all level pairs:
      - Mean Diff (posterior mean)
      - 95% HDI
      - P(G1>G2)
      - Cohen's d posterior using per-draw pooled SD: sqrt((v1 + v2)/2)
    """
    rows = []
    eps = 1e-12
    for i, j in itertools.combinations(range(len(levels)), 2):
        a, b = levels[i], levels[j]
        sa, sb = mean_draws[a], mean_draws[b]
        va, vb = var_draws[a],  var_draws[b]

        valid = (~np.isnan(sa)) & (~np.isnan(sb)) & (~np.isnan(va)) & (~np.isnan(vb))
        if not np.any(valid):
            rows.append({
                "Group 1": a, "Group 2": b,
                "Mean Diff (G1-G2)": np.nan,
                "HDI 2.5%": np.nan, "HDI 97.5%": np.nan,
                "P(G1>G2)": np.nan,
                "Cohen's d (median)": np.nan,
                "Cohen's d HDI 2.5%": np.nan,
                "Cohen's d HDI 97.5%": np.nan
            })
            continue

        diffs = sa[valid] - sb[valid]
        pooled_sd = np.sqrt(0.5 * (va[valid] + vb[valid]))
        d_draws = diffs / np.maximum(pooled_sd, eps)

        md = np.mean(diffs)
        dlo, dhi = hdi(diffs, 0.95)
        pgt = (diffs > 0).mean()
        d_med = np.median(d_draws)
        dlo_d, dhi_d = hdi(d_draws, 0.95)

        rows.append({
            "Group 1": a,
            "Group 2": b,
            "Mean Diff (G1-G2)": md,
            "HDI 2.5%": dlo,
            "HDI 97.5%": dhi,
            "P(G1>G2)": pgt,
            "Cohen's d (median)": d_med,
            "Cohen's d HDI 2.5%": dlo_d,
            "Cohen's d HDI 97.5%": dhi_d
        })
    return pd.DataFrame(rows)

def plot_posterior_means(levels, mean_draws, title, outfile):
    plt.figure(figsize=(7,5))
    for lvl in levels:
        plt.hist(mean_draws[lvl], bins=40, alpha=0.5, density=True, label=lvl)
    plt.xlabel("Posterior Mean Freezing (%)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_posterior_diffs(mean_draws, baseline, title, outfile):
    """NOW uses baseline − group for differences (signs align with your chosen convention)."""
    plt.figure(figsize=(7,5))
    baseline_draws = mean_draws[baseline]
    for lvl, draws in mean_draws.items():
        if lvl == baseline:
            continue
        diffs = baseline_draws - draws     # <-- flipped here
        plt.hist(diffs, bins=40, alpha=0.5, density=True, label=f"{baseline} - {lvl}")
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Posterior Difference (baseline − group)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_apoe3_baseline_comparisons(mean_draws, outfile, baseline='APOE3'):
    """Plot APOE diffs vs APOE3 and mark each group's 95% HDI on the x-axis (baseline − group)."""
    plt.figure(figsize=(7,5))
    baseline_draws = mean_draws[baseline]
    for lvl, draws in mean_draws.items():
        if lvl == baseline:
            continue
        diffs = baseline_draws - draws     # <-- flipped here
        lo, hi = hdi(diffs, 0.95)
        plt.hist(diffs, bins=40, alpha=0.5, density=True, label=f"{baseline} - {lvl}")
        # HDI segment at y≈0
        plt.plot([lo, hi], [0, 0], lw=4)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel(f"Posterior Difference (baseline − group)")
    plt.ylabel("Density")
    plt.title(f"Posterior Differences vs {baseline} (HDI marked)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

# ===== Load & prepare data =====
df = pd.read_csv(input_path)
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

if use_sound_filter and 'Component Name' in df.columns:
    m = df['Component Name'].str.startswith('Sound', na=False)
    if m.any():
        df = df[m].copy()

needed = set([dependent_var, 'Animal ID']) | set(factors)
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in set(['APOE', 'HN', 'Sex', 'Diet', 'Lifestyle', 'Component Name', 'Animal ID']):
    if col in df.columns:
        df[col] = df[col].astype('category')

df = df.dropna(subset=[dependent_var]).copy()

# ===== Run Bayesian bootstrap per factor =====
excel_out = os.path.splitext(output_path_freq)[0] + '_bayesian.xlsx'
plot_dir = os.path.dirname(excel_out)
os.makedirs(plot_dir, exist_ok=True)

with pd.ExcelWriter(excel_out, engine='openpyxl', mode='w') as writer:
    meta_info = pd.DataFrame({
        "Setting": ["B (draws)", "Seed", "Clustered by", "Filtered to Sound?"],
        "Value": [B, seed, "Animal ID", bool(use_sound_filter)]
    })
    meta_info.to_excel(writer, sheet_name='Meta', index=False)

    for factor in factors:
        print(f"Processing factor: {factor}")
        n_levels = df[factor].cat.categories.size
        if n_levels < 2:
            pd.DataFrame({"Warning": [f"Factor '{factor}' has <2 levels; skipping."]})\
                .to_excel(writer, sheet_name=f'{factor}_Summary', index=False)
            continue

        levels, mean_draws, var_draws = bayes_bootstrap_group_means_cluster(
            df=df,
            value_col=dependent_var,
            group_col=factor,
            cluster_col='Animal ID',
            B=B,
            rng=rng
        )

        # Summaries and pairwise (now with Cohen's d)
        summary_df = summarize_posterior_means(levels, mean_draws)
        comps_df = pairwise_comparisons(levels, mean_draws, var_draws)
        summary_df.to_excel(writer, sheet_name=f'{factor}_Summary', index=False)
        comps_df.to_excel(writer, sheet_name=f'{factor}_Comparisons', index=False)

        # Only plot & baseline sheet for APOE
        if factor == 'APOE':
            # Ensure APOE3 exists
            if apoe_baseline not in levels:
                print(f"⚠️ Baseline '{apoe_baseline}' not found in APOE levels {levels}. Using first level.")
                baseline = levels[0]
            else:
                baseline = apoe_baseline

            plot_posterior_means(levels, mean_draws, "Posterior Means by APOE",
                                 os.path.join(plot_dir, "APOE_posterior_means.png"))
            plot_posterior_diffs(mean_draws, baseline=baseline,
                                 title=f"Posterior Differences (baseline − group) vs {baseline}",
                                 outfile=os.path.join(plot_dir, f"APOE_posterior_diffs_vs_{baseline}.png"))
            plot_apoe3_baseline_comparisons(mean_draws,
                                 outfile=os.path.join(plot_dir, f"APOE_baseline_{baseline}_with_HDI.png"),
                                 baseline=baseline)

            # Dedicated sheet: comparisons vs APOE3 only (ordered)
            base_means = mean_draws[baseline]
            base_vars  = var_draws[baseline]
            rows = []
            eps = 1e-12
            for lvl in [l for l in levels if l != baseline]:
                # ---- flipped subtraction here to baseline − group ----
                diffs = base_means - mean_draws[lvl]
                dlo, dhi = hdi(diffs, 0.95)
                pgt = (diffs > 0).mean()
                pooled_sd = np.sqrt(0.5 * (base_vars + var_draws[lvl]))
                d_draws = diffs / np.maximum(pooled_sd, eps)
                d_lo, d_hi = hdi(d_draws, 0.95)
                rows.append({
                    "Group 1 (baseline)": baseline,
                    "Group 2": lvl,
                    "Mean Diff (Baseline - Group)": float(np.mean(diffs)),
                    "HDI 2.5%": float(dlo),
                    "HDI 97.5%": float(dhi),
                    "P(diff>0)": float(pgt),
                    "Cohen's d (median)": float(np.median(d_draws)),
                    "Cohen's d HDI 2.5%": float(d_lo),
                    "Cohen's d HDI 97.5%": float(d_hi),
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name='APOE_vs_APOE3', index=False)

print(f"✅ Results saved to: {excel_out}")
print(f"✅ Plots saved to: {plot_dir}")
