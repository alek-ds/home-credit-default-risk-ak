import pandas as pd
import numpy as np
import os
from typing import List, Literal, Optional
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_quantitative_distribution(
    df: pd.DataFrame,
    target: str,
    save_dir: Optional[str] = None
) -> None:
    """
    Creates visualization for a quantitative variable
    - histogram with mean and median lines
    - boxplot with mean and median lines
    """

    # Mean and median
    mean_val = df[target].mean()
    median_val = df[target].median()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Quantitative variable: {target}", fontsize=14, fontweight="bold")

    # Histogram
    sns.histplot(df[target].dropna(), kde=True, ax=axes[0], color="skyblue")
    axes[0].axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    axes[0].axvline(median_val, color="green", linestyle=":", label=f"Median = {median_val:.2f}")
    axes[0].set_title("Histogram")
    axes[0].legend()

    # Boxplot
    sns.boxplot(x=df[target], ax=axes[1], color="lightgray")
    axes[1].axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    axes[1].axvline(median_val, color="green", linestyle=":", label=f"Median = {median_val:.2f}")
    axes[1].set_title("Boxplot")
    axes[1].legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{target}_distribution.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.show()



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
