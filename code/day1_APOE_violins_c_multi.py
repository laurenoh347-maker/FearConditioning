import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load your data ===
input_path = "/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day1_filt_noKO.csv"
df = pd.read_csv(input_path)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# === Plot function ===
def plot_violin_box(data, x_var, y_var, hue_var, outpath):
    plt.figure(figsize=(8, 6))

    # Violin plot (distribution shape, transparent)
    sns.violinplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        split=True,
        inner=None,
        alpha=0.15,
        linewidth=0.8
    )

    # Overlay raw data (gray points)
    sns.stripplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        dodge=True,
        color="gray",
        alpha=0.5,
        size=3
    )

    # Boxplot on top (narrow, with whiskers + outliers)
    sns.boxplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        width=0.2,
        showcaps=True,
        boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.2, 'zorder':3},
        whiskerprops={'color':'black', 'linewidth':1.2},
        flierprops={'marker': 'o', 'color':'black', 'alpha':0.6, 'markersize':4},
        medianprops={'color':'black', 'linewidth':2, 'linestyle':'--'}
    )

    # Fix duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title=hue_var)

    # Labels and style
    plt.title(f"{x_var} by {hue_var} - Day 1")
    plt.ylabel(y_var)
    plt.xlabel(x_var)
    sns.despine()
    plt.tight_layout()

    # Save and show
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"✅ Saved: {outpath}")


# === Variables you want to plot ===
y_var = "Pct Total Time Freezing"
group_vars = ["APOE", "Diet", "Lifestyle", "HN"]

for g in group_vars:
    plot_violin_box(
        df,
        x_var=g,
        y_var=y_var,
        hue_var="Sex",   # always split by sex
        outpath=f"/Users/bass/Downloads/{g}_sex_day1_violin_box.png"
    )


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# === Load your data ===
input_path = "/Users/bass/Downloads/LaurenOh_FC/data/FC/FC_Day1_filt_noKO.csv"
df = pd.read_csv(input_path)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# === Residualize outcome by Age_months ===
y = df["Pct Total Time Freezing"]
X = sm.add_constant(df["Age_Months"])  # add intercept
model = sm.OLS(y, X, missing="drop").fit()
df["PctFreezing_resid"] = model.resid  # residualized outcome

print(model.summary())  # optional: check regression fit

# === Plot function ===
def plot_violin_box(data, x_var, y_var, hue_var, outpath, title_suffix=""):
    plt.figure(figsize=(8, 6))

    # Violin plot (distribution shape, transparent)
    sns.violinplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        split=True,
        inner=None,
        alpha=0.15,
        linewidth=0.8
    )

    # Overlay raw data (gray points)
    sns.stripplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        dodge=True,
        color="gray",
        alpha=0.5,
        size=3
    )

    # Boxplot on top
    sns.boxplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        width=0.2,
        showcaps=True,
        boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.2, 'zorder':3},
        whiskerprops={'color':'black', 'linewidth':1.2},
        flierprops={'marker': 'o', 'color':'black', 'alpha':0.6, 'markersize':4},
        medianprops={'color':'black', 'linewidth':2, 'linestyle':'--'}
    )

    # Fix duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title=hue_var)

    # Labels and style
    plt.title(f"{x_var} by {hue_var} - Day 1 {title_suffix}")
    plt.ylabel(y_var)
    plt.xlabel(x_var)
    sns.despine()
    plt.tight_layout()

    # Save and show
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"✅ Saved: {outpath}")


# === Variables you want to plot ===
group_vars = ["APOE", "Diet", "Lifestyle", "HN"]

for g in group_vars:
    # Raw outcome
    plot_violin_box(
        df,
        x_var=g,
        y_var="Pct Total Time Freezing",
        hue_var="Sex",
        outpath=f"/Users/bass/Downloads/{g}_sex_day1_violin_box.png",
        title_suffix="(Raw)"
    )
    
    # Residualized outcome
    plot_violin_box(
        df,
        x_var=g,
        y_var="PctFreezing_resid",
        hue_var="Sex",
        outpath=f"/Users/bass/Downloads/{g}_sex_day1_violin_box_resid.png",
        title_suffix="(Residualized by Age)"
    )
