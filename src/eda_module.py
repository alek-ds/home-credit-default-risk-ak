from __future__ import annotations
import pandas as pd
import numpy as np
import os
from typing import List, Literal, Optional, Any
from scipy.stats import spearmanr, mannwhitneyu, kruskal, ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


#=======================================================================
# Univariate analysis
def plot_quantitative_distribution(
        df: pd.DataFrame,
        quant_var: str,
        hist_bins: str | int = 'auto',
        log_scale: bool = False,
        show_outliers: bool = True,
        plot_violin: bool = False,
        save_dir: Optional[str] = None
) -> None:
    """ 
    Plots univariate distribution of a quantitative variable
    Creates:
        - histogram
        - boxplot / violinplot
    """

    # Checks
    if quant_var not in df.columns:
        raise KeyError(f"Column '{quant_var}' not found")
    if not pd.api.types.is_numeric_dtype(df[quant_var]):
        raise ValueError(f"Column '{quant_var}' has wrong dtype: {df[quant_var].dtype}")
    
    
    # Metrics
    nans = df[quant_var].isna().sum()
    nans_share = nans / df.shape[0]
    plot_data = df[[quant_var]].dropna().copy()
    mean = plot_data[quant_var].mean()
    median = plot_data[quant_var].median()

    if plot_data.empty:
        raise ValueError(f"Column '{quant_var}' contains only missing values.")
   
    if log_scale and (plot_data[quant_var] <= 0).any():
        raise ValueError(
            f"log_scale=True requires all non-missing values of '{quant_var}' to be > 0."
        )

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    sns.histplot(
        data=plot_data,
        x=quant_var,
        bins=hist_bins,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0]
    )
    axes[0].set_title(f"Histogram of {quant_var}")
    axes[0].set_xlabel(quant_var)
    axes[0].set_ylabel("Density")

    axes[0].axvline(mean, linestyle="--", linewidth=1.5, label=f"mean = {mean:.2f}")
    axes[0].axvline(median, linestyle=":", linewidth=1.5, label=f"median = {median:.2f}")
    axes[0].legend()

    axes[0].text(
        0.98,
        0.95,
        f"Missing: {nans} ({nans_share:.1%})",
        transform=axes[0].transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    # Boxplot or violiplot
    if plot_violin:
        sns.violinplot(
            data=plot_data,
            y=quant_var,
            ax=axes[1]
        )
        axes[1].set_title(f"Violinplot of {quant_var}")
    else:
        sns.boxplot(
            data=plot_data,
            y=quant_var,
            showfliers=show_outliers,
            ax=axes[1]
        )
        axes[1].set_title(f"Boxplot of {quant_var}")

    axes[1].set_ylabel(quant_var)

    # Log scale
    if log_scale:
        axes[0].set_xscale("log")
        axes[1].set_yscale("log")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{quant_var}_distribution.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_categorical_distribution(
        df: pd.DataFrame,
        cat_var: str,
        save_dir: Optional[str] = None,
        top_n: Optional[int] = None
) -> None:
    """ 
    Plots univariate distribution of a categorical variable
    Creates:
        - barplot of category counts
    """

    # Checks
    if cat_var not in df.columns:
        raise KeyError(f"Column '{cat_var}' not found")
    if not (
        pd.api.types.is_object_dtype(df[cat_var])
        or pd.api.types.is_categorical_dtype(df[cat_var])
    ):
        raise ValueError(f"Column '{cat_var}' has wrong dtype: {df[cat_var].dtype}")
    
    if top_n is not None and top_n <= 0:
        raise ValueError(f"top_n ({top_n}) must be a positive integer or None")

    # Misssing metrics 
    nans = df[cat_var].isna().sum()
    nans_share = nans / len(df)
    
    plot_data = df[cat_var].astype("object").fillna('Missing').copy()
    
    if plot_data.empty:
        raise ValueError(f"Column '{cat_var}' contains only missing values.")
    
    n_unique_non_missing = df[cat_var].nunique(dropna=True)
    value_counts = plot_data.value_counts(dropna=False)

    # Top-n handling
    if top_n is not None and len(value_counts) > top_n:
        missing_count = value_counts.get("Missing", 0)

        non_missing_counts = value_counts.drop(index="Missing", errors="ignore")

        if len(non_missing_counts) > top_n:
            top_counts = non_missing_counts.iloc[:top_n]
            other_count = non_missing_counts.iloc[top_n:].sum()

            final_counts = top_counts.copy()
            if other_count > 0:
                final_counts.loc["Other"] = other_count
        else:
            final_counts = non_missing_counts.copy()

        if missing_count > 0:
            final_counts.loc["Missing"] = missing_count
        
        value_counts = final_counts
    
    mode = value_counts.index[0]
    mode_count = value_counts.iloc[0]
    mode_share = mode_count / len(plot_data)
  
    # Plot
    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        x=value_counts.values,
        y=value_counts.index,
        ax=ax
    )

    ax.set_title(f"Distribution of {cat_var}")
    ax.set_xlabel(cat_var)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    ax.text(
        0.98,
        0.18,
        f"Missing: {nans} ({nans_share:.1%})\n"
        f"Unique categories: {n_unique_non_missing}\n"
        f"Mode: {mode} ({mode_share:.1%})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{cat_var}_distribution.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()



def plot_binary_distribution(
    df: pd.DataFrame,
    binary_var: str,
    save_dir: Optional[str] = None,
    facecolor: str = "#FFFFFF",
    colors: Optional[list[str]] = None
) -> None:
    """
    Plot univariate distribution of a binary variable as donut chart(s).

    Creates:
    - one donut if there are no missing values
    - two donuts if there are missing values:
        * excluding missing
        * including missing

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    binary_var : str
        Name of binary variable.
    save_dir : str | None, default=None
        Directory where plot should be saved.
    facecolor : str, default="#FFFFFF"
        Background color for figure and axes.
    colors : list[str] | None, default=None
        Colors for categories. If None, default matplotlib colors are used.
    """

    # Checks
    if binary_var not in df.columns:
        raise KeyError(f"Column '{binary_var}' not found")

    non_missing = df[binary_var].dropna()
    unique_vals = list(pd.unique(non_missing))

    if len(unique_vals) != 2:
        raise ValueError(
            f"Column '{binary_var}' must have exactly 2 unique non-missing values. "
            f"Found {len(unique_vals)}: {unique_vals}"
        )

    # Stable order
    if set(unique_vals) == {0, 1}:
        order = [0, 1]
    elif set(unique_vals) == {"N", "Y"}:
        order = ["N", "Y"]
    else:
        order = sorted(unique_vals)

    nans = df[binary_var].isna().sum()
    nans_share = nans / len(df)

    counts_no_nan = non_missing.value_counts().reindex(order)

    plot_series = df[binary_var].astype("object")
    plot_series = plot_series.where(plot_series.notna(), "Missing")

    counts_with_nan = plot_series.value_counts()

    counts_with_nan = counts_with_nan.reindex(
        order + (["Missing"] if nans > 0 else [])
    )

    # Default colors
    if colors is None:
        if nans > 0:
            colors = ["#4C72B0", "#E72121", "#9A9A9A"]
        else:
            colors = ["#4C72B0", "#E72121"]

    def make_legend_labels(counts: pd.Series) -> list[str]:
        total = counts.sum()
        return [
            f"{idx} — {val} ({val / total:.1%})"
            for idx, val in counts.items()
        ]

    def draw_donut(ax, counts: pd.Series, title: str, center_text: str, donut_colors: list[str]) -> None:
        wedges, _, autotexts = ax.pie(
            counts.values,
            labels=None,
            colors=donut_colors[:len(counts)],
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            pctdistance=0.78,
            wedgeprops=dict(width=0.38, edgecolor=facecolor, linewidth=3)
        )

        for autotext in autotexts:
            autotext.set_color("black" if facecolor == "#FFFFFF" else "white")
            autotext.set_fontsize(11)
            autotext.set_weight("bold")

        text_color = "black" if facecolor == "#FFFFFF" else "white"

        ax.text(
            0, 0.05, center_text,
            ha="center", va="center",
            fontsize=13, color=text_color, weight="bold"
        )
        ax.text(
            0, -0.08, title,
            ha="center", va="center",
            fontsize=10, color=text_color
        )

        ax.set_title(title, fontsize=14, color=text_color, weight="bold", pad=16)

        legend = ax.legend(
            wedges,
            make_legend_labels(counts),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=10,
            title="Categories",
            title_fontsize=11
        )
        plt.setp(legend.get_texts(), color=text_color)
        plt.setp(legend.get_title(), color=text_color)

        ax.set_facecolor(facecolor)

    # Figure layout
    if nans > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=facecolor)
        axes = axes.ravel()

        draw_donut(
            ax=axes[0],
            counts=counts_no_nan,
            title="Excluding missing",
            center_text=f"Distribution of \n{binary_var}",
            donut_colors=colors
        )

        draw_donut(
            ax=axes[1],
            counts=counts_with_nan,
            title="Including missing",
            center_text=f"Distribution of \n{binary_var}",
            donut_colors=colors
        )

        #title_text = f"Distribution of {binary_var} | Missing: {nans} ({nans_share:.1%})"
    else:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=facecolor)

        draw_donut(
            ax=ax,
            counts=counts_no_nan,
            #title="Binary distribution",
            title='',
            center_text=f"Distribution of \n{binary_var}",
            donut_colors=colors
        )

        #title_text = f"Distribution of {binary_var}"

    # fig.suptitle(
    #     title_text,
    #     fontsize=16,
    #     color="black" if facecolor == "#FFFFFF" else "white",
    #     weight="bold",
    #     y=0.98
    # )

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{binary_var}_binary_distribution.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    plt.show()


def plot_binary_distribution2(
        df: pd.DataFrame,
        binary_var: str,
        save_dir: Optional[str] = None
) -> None:
    """
    Plots univariate distribution of a binary variable
    Creates:
    - donut plot excluding missing values
    - donut plot including missing values as a separate category
    """

    # Checks
    if binary_var not in df.columns:
        raise KeyError(f"Column '{binary_var}' not found")
    
    non_missing = df[binary_var].dropna()
    unique_vals = list(pd.unique(non_missing))

    if len(unique_vals) != 2:
        raise ValueError(
            f"Column '{binary_var}' must have exactly 2 unique non-missing values. "
            f"Found {len(unique_vals)}: {unique_vals}"
        )
    
        # Add ordering logic here
    if set(unique_vals) == {0, 1}:
        order = [0, 1]
    elif set(unique_vals) == {"N", "Y"}:
        order = ["N", "Y"]
    else:
        order = sorted(unique_vals)
    
    
    # Counts excluding missing
    counts_no_nan = non_missing.value_counts().reindex(order)
    
    # Counts including missing
    counts_with_nan = df[binary_var].astype("object").fillna("Missing").value_counts()

    nans = df[binary_var].isna().sum()
    nans_share = nans/ len(df)

    fig, axes = plt.subplots(1,2, figsize=(12,6))

    # Donut 1: excluding missing
    axes[0].pie(
        counts_no_nan.values,
        labels=counts_no_nan.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    axes[0].set_title(f"{binary_var}\n(excluding missing)")

    # Donut 2: including missing
    axes[1].pie(
        counts_with_nan.values,
        labels=counts_with_nan.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    axes[1].set_title(f"{binary_var}\n(including missing)")

    fig.suptitle(
        f"Distribution of {binary_var} | Missing: {nans} ({nans_share:.1%})",
        fontsize=13
    )

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{binary_var}_binary_distribution.png"),
            dpi=300,
            bbox_inches='tight'
        )
    
    plt.show()



#=======================================================================
# Bivariate analysis
def plot_quantitative_vs_binary(
    df: pd.DataFrame,
    quant_var: str,
    binary_var: str,
    hist_bins: str | int = "auto",
    save_dir: Optional[str] = None
) -> None:
    """
    Quick EDA function:
    - histogram by binary group
    - boxplot by binary group
    - Mann-Whitney U test
    - group medians displayed on plot
    """

    # Keep relevant columns and drop missing values
    data = df[[quant_var, binary_var]].dropna().copy()

    # Checks
    if quant_var not in df.columns:
        raise KeyError(f"Column '{quant_var}' not found.")
    if binary_var not in df.columns:
        raise KeyError(f"Column '{binary_var}' not found.")
    if data[binary_var].nunique() != 2:
        raise ValueError(f"'{binary_var}' must have exactly 2 non-null unique values.")

    # Order groups
    groups = sorted(data[binary_var].unique())
    if set(groups) == {0, 1}:
        groups = [0, 1]

    g1, g2 = groups

    x1 = data.loc[data[binary_var] == g1, quant_var]
    x2 = data.loc[data[binary_var] == g2, quant_var]

    # Metrics
    median1 = x1.median()
    median2 = x2.median()
    median_diff = median2 - median1

    stat, p_value = mannwhitneyu(x1, x2, alternative="two-sided")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(
        data=data,
        x=quant_var,
        hue=binary_var,
        hue_order=groups,
        bins=hist_bins,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0]
    )
    axes[0].set_title(f"Histogram of {quant_var}")
    axes[0].set_xlabel(quant_var)
    axes[0].set_ylabel("Density")

    axes[0].axvline(median1, linestyle="--", linewidth=1.5, label=f"{g1} median = {median1:.2f}")
    axes[0].axvline(median2, linestyle=":", linewidth=1.5, label=f"{g2} median = {median2:.2f}")
    axes[0].legend()

    axes[0].text(
        0.98,
        0.95,
        f"Mann-Whitney U p-value = {p_value:.3e}\nMedian diff ({g2}-{g1}) = {median_diff:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    sns.boxplot(
        data=data,
        x=binary_var,
        y=quant_var,
        order=groups,
        ax=axes[1]
    )
    axes[1].set_title(f"Boxplot of {quant_var} by {binary_var}")
    axes[1].set_xlabel(binary_var)
    axes[1].set_ylabel(quant_var)

    fig.suptitle(f"{quant_var} vs {binary_var}", fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{quant_var}_vs_{binary_var}.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_binary_vs_binary(
        df: pd.DataFrame,
        binary_var: str,
        target_var: str,
        save_dir: Optional[str] = None
) -> None:
    """
    Plots relationship between a binary variable and a binary target.

    Creates:
    - 100% stacked bar plot

    Shows:
    - chi-square p-value
    - difference in target rate across categories ob binary_var
    """

    # Checks
    for col in [binary_var, target_var]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found")
    
    plot_data = df[[binary_var, target_var]].dropna().copy()

    if plot_data.empty:
        raise ValueError("No complete cases available after dropping missing values")
    
    for col in [binary_var, target_var]:
        n_unique = plot_data[col].nunique()
        if n_unique != 2:
            raise ValueError(
                f"Column '{col}' must have exactly 2 unique non-missing values. Found {n_unique}."
            )
        
    def get_binary_order(series: pd.Series) -> list[Any]:
        unique_vals = list(pd.unique(series))

        if set(unique_vals) == {0, 1}:
            return [0, 1]
        elif set(unique_vals) == {"N", "Y"}:
            return ["N", "Y"]
        else: 
            return sorted(unique_vals)
        
    binary_order = get_binary_order(plot_data[binary_var])
    target_order = get_binary_order(plot_data[target_var])

    # Contigency table
    counts = pd.crosstab(plot_data[binary_var], plot_data[target_var])
    counts = counts.reindex(index=binary_order, columns=target_order, fill_value=0)

    proportions = counts.div(counts.sum(axis=1), axis=0)

    # Metrics
    chi2, p_value, _, _ = chi2_contingency(counts)

    positive_class = target_order[1]
    target_rates = proportions[positive_class]
    rate_diff_pp = (target_rates.iloc[1] - target_rates.iloc[0]) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = proportions.index.astype(str)
    x_pos = range(len(x_labels))

    bottom = pd.Series(0.0, index=proportions.index)

    class_colors = {
        target_order[0]: "steelblue",
        target_order[1]: "red"
    }

    for target_class in target_order:
        heights = proportions[target_class]

        ax.bar(
            x_pos,
            heights,
            bottom=bottom,
            label=f"{target_var} = {target_class}",
            color=class_colors[target_class]
        )

        # Add percentage labels inside bar segments
        for i, (idx, height) in enumerate(heights.items()):
            if height > 0.03:
                ax.text(
                    i,
                    bottom.loc[idx] + height / 2,
                    f"{height:.1%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold"
                )

        bottom += heights

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels)

    ax.set_title(f"{binary_var} vs {target_var}")
    ax.set_xlabel(binary_var)
    ax.set_ylabel("")
    ax.set_ylim(0,1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(title='')

    ax.text(
        0.98,
        0.11,
        f"Chi-square p_value = {p_value:.3e}\n"
        f"{target_var}={positive_class} rate diff = {rate_diff_pp:.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{binary_var}_vs_{target_var}.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()



def plot_categorical_vs_binary(
    df: pd.DataFrame,
    cat_var: str,
    target_var: str,
    top_n: Optional[int] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Plot relationship between a categorical variable and a binary target.

    Creates:
    - 100% stacked bar plot

    Shows:
    - chi-square p-value
    - range of target rate across categories

    Notes
    -----
    Missing values in cat_var are treated as a separate category: 'Missing'.
    If top_n is provided, only the top_n most frequent non-missing categories
    are shown, remaining non-missing categories are grouped as 'Other'.
    """

    # Checks
    for col in [cat_var, target_var]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found")

    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be a positive integer or None")

    # Target must be binary (ignoring NaNs)
    target_non_missing = df[target_var].dropna()
    if target_non_missing.nunique() != 2:
        raise ValueError(
            f"Column '{target_var}' must have exactly 2 unique non-missing values. "
            f"Found {target_non_missing.nunique()}."
        )

    def get_binary_order(series: pd.Series) -> list[Any]:
        unique_vals = list(pd.unique(series.dropna()))
        if set(unique_vals) == {0, 1}:
            return [0, 1]
        elif set(unique_vals) == {"N", "Y"}:
            return ["N", "Y"]
        else:
            return sorted(unique_vals)

    target_order = get_binary_order(df[target_var])

    # Keep only rows with non-missing target
    plot_data = df[[cat_var, target_var]].copy()
    plot_data = plot_data[plot_data[target_var].notna()].copy()

    if plot_data.empty:
        raise ValueError("No rows with non-missing target available.")

    # Treat missing categorical values as category
    plot_data[cat_var] = plot_data[cat_var].astype("object")
    plot_data[cat_var] = plot_data[cat_var].where(plot_data[cat_var].notna(), "Missing")

    # Apply top_n logic while preserving Missing
    cat_counts = plot_data[cat_var].value_counts()

    if top_n is not None:
        missing_count = cat_counts.get("Missing", 0)
        non_missing_counts = cat_counts.drop(index="Missing", errors="ignore")

        if len(non_missing_counts) > top_n:
            top_counts = non_missing_counts.iloc[:top_n]
            other_count = non_missing_counts.iloc[top_n:].sum()

            kept_categories = list(top_counts.index)
            plot_data.loc[~plot_data[cat_var].isin(kept_categories + ["Missing"]), cat_var] = "Other"

    # Recompute counts after possible grouping
    cat_counts = plot_data[cat_var].value_counts()

    # Category order
    if "Missing" in cat_counts.index:
        non_missing_order = [cat for cat in cat_counts.index if cat != "Missing"]
        category_order = non_missing_order + ["Missing"]
    else:
        category_order = list(cat_counts.index)

    # Contingency table
    counts = pd.crosstab(plot_data[cat_var], plot_data[target_var])
    counts = counts.reindex(index=category_order, columns=target_order, fill_value=0)

    proportions = counts.div(counts.sum(axis=1), axis=0)

    # Metrics
    chi2, p_value, _, _ = chi2_contingency(counts)

    positive_class = target_order[1]
    target_rates = proportions[positive_class]
    rate_range_pp = (target_rates.max() - target_rates.min()) * 100

    # Colors
    class_colors = {
        target_order[0]: "steelblue",
        target_order[1]: "red"
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = proportions.index.astype(str)
    x_pos = range(len(x_labels))

    bottom = pd.Series(0.0, index=proportions.index)

    for target_class in target_order:
        heights = proportions[target_class]

        ax.bar(
            x_pos,
            heights,
            bottom=bottom,
            label=f"{target_var} = {target_class}",
            color=class_colors[target_class]
        )

        # Add percentage labels inside bar segments
        for i, (idx, height) in enumerate(heights.items()):
            if height > 0.03:
                ax.text(
                    i,
                    bottom.loc[idx] + height / 2,
                    f"{height:.1%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold"
                )

        bottom += heights

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_title(f"{cat_var} vs {target_var}")
    ax.set_xlabel(cat_var)
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(title=target_var)

    ax.text(
        0.98,
        0.12,
        f"Chi-square p = {p_value:.3e}\n"
        f"{target_var}={positive_class} rate range = {rate_range_pp:.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{cat_var}_vs_{target_var}.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

#=======================================================================
# Multivariate analysis


















def filter_low_correlation(
    df: pd.DataFrame,
    target: str,
    method: Literal["spearman", "pearson"] = "spearman",
    threshold: float = 0.1,
    drop_target: bool = True
) -> List[str]:
    """
    Filtruje zmienne numeryczne o niskiej korelacji z wartością docelową (np. 'price').
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if target not in numeric_cols:
        raise ValueError(f"Kolumna docelowa '{target}' musi być zmienną numeryczną.")

    corr_matrix = df[numeric_cols].corr(method=method)
    corr_with_target = corr_matrix[target].abs().sort_values(ascending=False)
    strong_features = corr_with_target[abs(corr_with_target) >= threshold].index.tolist()

    if drop_target and target in strong_features:
        strong_features.remove(target)

    return strong_features

def show_correlations_with_target(
        df: pd.DataFrame,
        target: str,
        sort_by: Literal["pearson", "spearman", "kendall"] = "pearson",
        sorting: Literal["decreasing", "absolute_values"] = "decreasing",
        drop_target: bool = True,
        visualize: bool = True
) -> Optional[pd.DataFrame]:
    """
    Oblicza korelacje numerycznych zmiennych z kolumną docelową
    (Pearson, Spearman, Kendall) i zwraca tabelę posortowaną
    według wybranej metody. Opcjonalnie wyświetla heatmapę korelacji.
    """
    if any(df[target].isna()):
        print(f"{target} zawiera wartości NaN")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if target not in numeric_cols:
        raise ValueError(f"Kolumna docelowa '{target}' musi być zmienną numeryczną.")
    
    pearson_corr_matrix = df[numeric_cols].corr(method='pearson')
    spearman_corr_matrix = df[numeric_cols].corr(method='spearman')
    kendall_corr_matrix = df[numeric_cols].corr(method='kendall')
    
    
    corr_dfs = [pearson_corr_matrix, spearman_corr_matrix, kendall_corr_matrix]
    #corr_dict = {key, value for key in ['spearman', 'pearson', 'kendall'] for value in dataframe['price'] for dataframe in corr_dfs}
    
    keys = ['pearson', 'spearman', 'kendall']
    values = [dataframe[target] for dataframe in corr_dfs]
    corr_dict = {k: v for (k,v) in zip(keys, values)}
    corr_df = pd.DataFrame(corr_dict)
    if sorting == "decreasing":
        corr_df = corr_df.sort_values(by=sort_by, ascending=False)
    elif sorting == "absolute_values":
        corr_df = (
            corr_df
            .assign(abs_val=lambda x: x[sort_by].abs())
            .sort_values(by="abs_val", ascending=False)
            .drop(columns="abs_val")
        )
    else:
        print("Nieprawdiłowy sposób sortowania")
        return None
        
    if drop_target:
        corr_df = corr_df.drop(index=target, errors="ignore")

    if visualize and not corr_df.empty:
        #top_features = corr_df.head(15)
        #plt.figure(figsize=(6, max(6, len(top_features) * 0.4)))
        plt.figure(figsize=(6, max(6, len(corr_df) * 0.4)))
        sns.heatmap(
            #top_features[[sort_by]].T,
            corr_df,
            annot=True,
            cmap="coolwarm",
            center=0,
            cbar=True,
            fmt=".2f",
            linewidths=0.5
        )
        plt.title(f"Korelacja (sortowanie: {sort_by.capitalize()}, {sorting}) z '{target}'", fontsize=12, weight="bold")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return None
    
    return corr_df




def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca tabelę z liczbą i procentem braków danych w kolumnach DataFrame.
    """
    summary = (
        df.isna().sum()
        .to_frame("missing_count")
        .assign(missing_pct=lambda x: x["missing_count"] / len(df) * 100)
        .sort_values("missing_pct", ascending=False)
    )
    return summary.query("missing_count > 0")


def test_binary_features(
    df: pd.DataFrame,
    target: str,
    alpha: float = 0.05,
    min_group_size: int = 5,
    verbose: bool = False
) -> List[str]:
    """
    Zwraca listę binarnych zmiennych (0/1), które mają istotny wpływ na zmienną docelową
    (np. 'price'), testując różnice przy użyciu testu Manna–Whitneya.
    Pomija zmienne, gdzie którakolwiek grupa ma mniej niż min_group_size obserwacji.
    """
    significant_features = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dropna().nunique() == 2:
            group_0 = df.loc[df[col] == 0, target].dropna()
            group_1 = df.loc[df[col] == 1, target].dropna()

            # sprawdzenie liczności
            if len(group_0) < min_group_size or len(group_1) < min_group_size:
                if verbose:
                    print(
                        f"[POMINIĘTO] {col}: zbyt mało danych "
                        f"(grupa 0: {len(group_0)}, grupa 1: {len(group_1)})"
                    )
                continue

            stat, p = mannwhitneyu(group_0, group_1, alternative="two-sided")
            if p < alpha:
                significant_features.append(col)
                if verbose:
                    print(f"[OK] {col}: p={p:.5f}")
            elif verbose:
                print(f"[NIEISTOTNE] {col}: p={p:.5f}")

    return significant_features


def test_categorical_features(
    df: pd.DataFrame,
    target: str,
    alpha: float = 0.05,
    min_group_size: int = 5,
    verbose: bool = False
) -> List[str]:
    """
    Zwraca listę kategorycznych zmiennych, które mają istotny wpływ na zmienną docelową
    (np. 'price'), testując różnice przy użyciu testu Kruskala–Wallisa.
    Pomija zmienne, gdzie którakolwiek kategoria ma mniej niż min_group_size obserwacji.

    """
    significant_features = []
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ["object", "category"]]

    for col in cat_cols:
        valid_groups = [
            df.loc[df[col] == val, target].dropna()
            for val in df[col].unique()
            if df.loc[df[col] == val, target].notna().sum() >= min_group_size
        ]

        # pomijamy, jeśli mniej niż 2 sensowne grupy
        if len(valid_groups) < 2:
            if verbose:
                print(f"[POMINIĘTO] {col}: mniej niż 2 grupy spełniające min_group_size={min_group_size}")
            continue

        stat, p = kruskal(*valid_groups)
        if p < alpha:
            significant_features.append(col)
            if verbose:
                print(f"[OK] {col}: p={p:.5f}")
        elif verbose:
            print(f"[NIEISTOTNE] {col}: p={p:.5f}")

    return significant_features


sns.set(style="whitegrid")


def plot_numeric_features(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Generuje wizualizacje dla zmiennych numerycznych:
    - histogram rozkładu zmiennej
    - scatterplot zmiennej względem targetu (np. 'price')
    """
    if features is None:
        features = df.select_dtypes(include=["float64", "int64"]).columns.drop(target)

    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Zmienna numeryczna: {col}", fontsize=14, fontweight="bold")

        # Histogram
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title("Rozkład wartości")
        axes[0].set_xlabel(col)

        # Scatter z targetem
        sns.scatterplot(x=df[col], y=df[target], alpha=0.6, ax=axes[1])
        axes[1].set_title(f"{col} vs {target}")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel(target)

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{col}_numeric.png"))
            plt.close(fig)
        else:
            plt.show()



def plot_binary_features(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Generuje wizualizacje dla zmiennych binarnych:
    - skumulowany wykres słupkowy udziałów 0/1
    - boxplot rozkładu targetu względem wartości binarnej
    """
    if features is None:
        features = [col for col in df.columns if df[col].dropna().nunique() == 2 and col != target]

    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Zmienna binarna: {col}", fontsize=14, fontweight="bold")

        # Wykres udziałów
        counts = df[col].value_counts(normalize=True).sort_index()
        axes[0].bar(counts.index.astype(str), counts.values, color=["#4CAF50", "#F44336"])
        axes[0].set_title("Udział wartości (0/1)")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Udział")

        # Boxplot zmiennej ilościowej
        sns.boxplot(x=col, y=target, data=df, ax=axes[1])
        axes[1].set_title(f"Rozkład {target} względem {col}")

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{col}_binary.png"))
            plt.close(fig)
        else:
            plt.show()

def plot_categorical_features(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Generuje wizualizacje dla zmiennych kategorycznych:
    - wykres słupkowy liczebności kategorii
    - boxplot rozkładu targetu względem kategorii
    """
    if features is None:
        features = [c for c in df.columns if str(df[c].dtype) in ["object", "category"] and df[c].nunique()>2]

    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Zmienna kategoryczna: {col}", fontsize=14, fontweight="bold")

        # Rozkład kategorii
        sns.countplot(x=col, data=df, ax=axes[0], order=df[col].value_counts().index)
        axes[0].set_title("Rozkład kategorii")
        axes[0].tick_params(axis='x', rotation=45)

        # Rozkład cen względem kategorii
        sns.boxplot(x=col, y=target, data=df, ax=axes[1], order=df[col].value_counts().index)
        axes[1].set_title(f"{target} względem {col}")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{col}_categorical.png"))
            plt.close(fig)
        else:
            plt.show()



