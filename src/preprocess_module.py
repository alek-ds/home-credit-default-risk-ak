import operator
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional


def _build_invalid_mask(
    series: pd.Series,
    invalid_value=None,
    invalid_condition=None
) -> pd.Series:
    """
    Build boolean mask of invalid values.

    Supported:
    - invalid_value=365243
    - invalid_condition='< 0'
    - invalid_condition='<= -1'
    - invalid_condition='== 365243'
    - invalid_condition=lambda s: s < 0
    """

    mask = pd.Series(False, index=series.index)

    if invalid_value is not None:
        if isinstance(invalid_value, (list, tuple, set, np.ndarray, pd.Series)):
            mask |= series.isin(invalid_value)
        else:
            mask |= series.eq(invalid_value)

    if invalid_condition is not None:
        if callable(invalid_condition):
            cond_mask = invalid_condition(series)
            if not isinstance(cond_mask, pd.Series):
                cond_mask = pd.Series(cond_mask, index=series.index)
            mask |= cond_mask.fillna(False)

        elif isinstance(invalid_condition, str):
            condition = invalid_condition.strip()

            ops = {
                "<": operator.lt,
                "<=": operator.le,
                ">": operator.gt,
                ">=": operator.ge,
                "==": operator.eq,
                "!=": operator.ne,
            }

            matched_op = None
            for op_symbol in ["<=", ">=", "==", "!=", "<", ">"]:
                if condition.startswith(op_symbol):
                    matched_op = op_symbol
                    threshold_str = condition[len(op_symbol):].strip()
                    break

            if matched_op is None:
                raise ValueError(
                    f"Unsupported invalid_condition: {invalid_condition!r}. "
                    "Use forms like '< 0', '== 365243', '>= 1000', "
                    "or pass a callable."
                )

            try:
                threshold = float(threshold_str)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse threshold in invalid_condition: {invalid_condition!r}"
                ) from e

            mask |= ops[matched_op](series, threshold).fillna(False)

        else:
            raise TypeError(
                "invalid_condition must be either a string, a callable, or None."
            )

    return mask.fillna(False)


def create_imputed_quantitative_features(
        df: pd.DataFrame,
        value_col: str,
        specs: dict,
        invalid_value=None,
        invalid_condition=None,
        add_invalid_flag: bool = True,
        add_clean_col: bool = True,
        return_summary: bool = True
):
    """
    Create multiple imputed versions of one variable using grouped/hierarchical rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Column to impute.
    specs : dict
        Dictionary where keys are output column names and values are dicts with:
            - method: 'median' or 'mean'
            - group_levels: list of grouping-column lists
              Example:
                  []
                  [['NAME_INCOME_TYPE']]
                  [['NAME_INCOME_TYPE', 'CODE_GENDER'], ['NAME_INCOME_TYPE']]
    invalid_value : scalar or iterable, optional
        Exact value(s) treated as invalid.
    invalid_condition : str or callable, optional
        Broader invalid rule, e.g. '< 0', '== 365243',
        or lambda s: (s < 0) | (s == 365243)
    add_invalid_flag : bool
        Whether to create {value_col}_invalid.
    add_clean_col : bool
        Whether to create {value_col}_clean.
    return_summary : bool
        Whether to return imputation diagnostics.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with added columns.
    summary_df : pd.DataFrame, optional
        Diagnostics for each created imputed column.
    """

    df_out = df.copy()

    clean_col = f"{value_col}_clean"
    invalid_flag_col = f"{value_col}_invalid"

    invalid_mask = _build_invalid_mask(
        df_out[value_col],
        invalid_value=invalid_value,
        invalid_condition=invalid_condition
    )

    if add_invalid_flag:
        df_out[invalid_flag_col] = invalid_mask.astype(int)

    df_out[clean_col] = df_out[value_col].mask(invalid_mask, np.nan)

    summary_rows = []

    for output_col, config in specs.items():
        method = config.get("method", "median")
        group_levels = config.get("group_levels", [])

        if method not in {"median", "mean"}:
            raise ValueError(
                f"Unsupported method {method!r} for {output_col}. Use 'median' or 'mean'."
            )
        
        df_out[output_col] = df_out[clean_col].copy()

        step_counts = []
        missing_before = df_out[output_col].isna().sum()

        for step_idx, group_cols in enumerate(group_levels, start=1):
            if not group_cols:
                continue

            fill_values = (
                df_out.groupby(group_cols, dropna=False)[clean_col]
                .agg(method)
                .rename("_fill_value")
                .reset_index()
            )

            df_out = df_out.merge(fill_values, on=group_cols, how="left")

            na_before_step = df_out[output_col].isna()
            df_out[output_col] = df_out[output_col].fillna(df_out["_fill_value"])
            filled_now = na_before_step.sum() - df_out[output_col].isna().sum()

            step_counts.append({
                "output_col": output_col,
                "step": step_idx,
                "group_level": tuple(group_cols),
                "filled_count": int(filled_now),
            })

            df_out = df_out.drop(columns="_fill_value")

        global_fill = getattr(df_out[clean_col], method)()
        na_before_global = df_out[output_col].isna().sum()
        df_out[output_col] = df_out[output_col].fillna(global_fill)
        global_filled = na_before_global - df_out[output_col].isna().sum()

        summary_rows.append({
            "output_col": output_col,
            "method": method,
            "n_invalid": int(invalid_mask.sum()),
            "missing_before_imputation": int(missing_before),
            "filled_by_hierarchy": int(sum(row["filled_count"] for row in step_counts)),
            "filled_by_global": int(global_filled),
            "missing_after_imputation": int(df_out[output_col].isna().sum()),
            "group_levels": str(group_levels),
        })

        summary_rows.extend(step_counts)

    if not add_clean_col:
        df_out = df_out.drop(columns=clean_col)

    summary_df = pd.DataFrame(summary_rows)

    if return_summary:
        return df_out, summary_df
    return df_out


def filter_binary_features(
    df: pd.DataFrame,
    binary_cols: List[str],
    minimum_share: float = 0.05,
    return_homogeneous_cols: bool = False,
    dropna: bool = True,
    return_summary: bool = True
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Filter binary features by the minimum share of the less frequent category.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    binary_cols : List[str]
        List of columns expected to be binary.
    minimum_share : float, default=0.05
        Minimum allowed share of the less frequent category.
        Columns with smaller minority share are treated as homogeneous.
    return_homogeneous_cols : bool, default=False
        If False, return columns that pass the filter.
        If True, return columns that fail the filter.
    dropna : bool, default=True
        Whether to exclude missing values when calculating category shares.
    return_summary : bool, default=True
        Whether to also return a summary dataframe.

    Returns
    -------
    List[str]
        Filtered list of column names.

    or

    Tuple[List[str], pd.DataFrame]
        Filtered list of column names and summary dataframe.
    """

    if not 0 <= minimum_share <= 0.5:
        raise ValueError("minimum_share must be between 0 and 0.5")

    missing_cols = [col for col in binary_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found: {missing_cols}")

    homogeneous_cols = []
    heterogeneous_cols = []
    summary_rows = []

    for col in binary_cols:
        value_counts = df[col].value_counts(dropna=dropna)
        n_unique = len(value_counts)

        if n_unique < 2:
            minority_category = value_counts.index[0] if n_unique == 1 else None
            minority_share = 0.0
            keep = False
            homogeneous_cols.append(col)

        elif n_unique > 2:
            raise ValueError(
                f"Column '{col}' is not binary. Found {n_unique} unique values."
            )

        else:
            minority_category = value_counts.idxmin()
            minority_share = value_counts.min() / value_counts.sum()
            keep = minority_share >= minimum_share

            if keep:
                heterogeneous_cols.append(col)
            else:
                homogeneous_cols.append(col)

        summary_rows.append({
            "feature": col,
            "n_unique_observed": n_unique,
            "minority_category": minority_category,
            "minority_share": minority_share,
            "minimum_share": minimum_share,
            "keep": keep
        })

    selected_cols = homogeneous_cols if return_homogeneous_cols else heterogeneous_cols

    if return_summary:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["keep", "minority_share", "feature"],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        return selected_cols, summary_df

    return selected_cols


def filter_high_nans_cols(
        df: pd.DataFrame,
        max_nans_share: float = 0.45,
        return_dropped_cols: bool = False,
        return_summary: bool = True
) -> Union[
    pd.DataFrame,
    Tuple[pd.DataFrame, List[str]],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, List[str], pd.DataFrame]
]:
    
    """
    Filter columns by maximum allowed share of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_nans_share : float, default=0.45
        Maximum allowed share of NaN values in a column.
        Columns with missing share greater than this threshold are dropped.
    return_dropped_cols : bool, default=False
        Whether to also return the list of dropped columns.
    return_summary : bool, default=True
        Whether to also return a summary dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Optionally also returns:
    - List[str]: dropped columns
    - pd.DataFrame: summary dataframe
    """

    if not 0 <= max_nans_share <= 1:
        raise ValueError("max_nan_share must be between 0.01 and 0.99")
    
    nan_share = df.isna().mean()

    kept_cols = nan_share[nan_share <= max_nans_share].index.tolist()
    dropped_cols = nan_share[nan_share > max_nans_share].index.tolist()

    filtered_df = df[kept_cols].copy()

    summary_df = pd.DataFrame({
        "feature": df.columns,
        "nan_share": nan_share.values,
        "max_nans_share": max_nans_share,
        "keep": [col in kept_cols for col in df.columns]
    }).sort_values(
        by = ["keep", "nan_share", "feature"],
        ascending = [True, False, True]
    ).reset_index(drop=True)

    outputs = [filtered_df]

    if return_dropped_cols:
        outputs.append(dropped_cols)

    if return_summary:
        outputs.append(summary_df)
    
    if len(outputs) == 1:
        return outputs[0]
    
    return tuple(outputs)


def drop_obs_with_nans_in_low_nan_cols(
    df: pd.DataFrame,
    max_nan_share: float = 0.05,
    return_filtering_cols_summary: bool = False,
    return_remaining_nan_cols_summary: bool = False,
    return_row_summary: bool = True
) -> Union[
    pd.DataFrame,
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Drop observations with missing values in columns whose NaN share is
    greater than 0 and less than or equal to max_nan_share.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_nan_share : float, default=0.05
        Maximum NaN share for a column to be treated as 'low-missing'.
        Rows with NaNs in such columns are dropped.
    return_dropped_cols_summary : bool, default=False
        Whether to return NaN shares for columns used to filter rows.
    return_kept_cols_summary : bool, default=False
        Whether to return NaN shares of columns with remaining NaNs after filtering.
    return_row_summary : bool, default=True
        Whether to return summary of rows affected by the function.

    Returns
    -------
    pd.DataFrame
        Dataframe after dropping observations.

    Optionally also returns
    -----------------------
    pd.Series
        NaN shares of low-missing columns used for row filtering.
    pd.Series
        NaN shares of columns still containing NaNs after filtering.
    pd.Series
        Summary of number and share of rows dropped.
    """

    if df.shape[0] == 0:
        raise ValueError("Input dataframe has no rows.")
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")
    
    if not 0 <= max_nan_share <= 1:
        raise ValueError("max_nan_share must be between 0 and 1")

    nan_shares = df.isna().mean()
    cols_used_for_filtering = nan_shares[(nan_shares > 0) & (nan_shares <= max_nan_share)]

    df_out = df.dropna(axis=0, subset=cols_used_for_filtering.index)

    remaining_nan_shares = df_out.isna().mean()
    remaining_nan_shares = remaining_nan_shares[remaining_nan_shares > 0]

    n_rows_before = int(len(df))
    n_rows_after = int(len(df_out))
    n_rows_dropped = n_rows_before - n_rows_after
    rows_dropped_share = n_rows_dropped / n_rows_before

    row_summary = pd.Series({
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_rows_dropped": n_rows_dropped,
        "rows_dropped_share": rows_dropped_share
    })

    outputs = [df_out]

    if return_filtering_cols_summary:
        outputs.append(cols_used_for_filtering.sort_values())

    if return_remaining_nan_cols_summary:
        outputs.append(remaining_nan_shares.sort_values())

    if return_row_summary:
        outputs.append(row_summary)

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)


def trim_quantitative_var(
    df: pd.DataFrame,
    quant_var: str,
    lower: float = 0.025,
    upper: float = 0.975,
    return_trimmed_df: bool = False,
    return_summary: bool = False
) -> Union[
    pd.Series,
    Tuple[pd.Series, pd.DataFrame],
    Tuple[pd.Series, pd.DataFrame, pd.Series]
]:
    """
    Drop observations outside the [lower, upper] quantile interval
    of a quantitative variable.

    By default the interval is [2.5%, 97.5%].
    Rows with missing values in quant_var are also excluded.
    """

    if df.shape[0] == 0:
        raise ValueError("Input dataframe has no rows.")
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")
    if quant_var not in df.columns:
        raise KeyError(f"Column '{quant_var}' not found.")
    if not 0 <= lower < upper <= 1:
        raise ValueError("Require 0 <= lower < upper <= 1.")
    
    df = df.dropna(subset=[quant_var])

    lower_bound = df[quant_var].quantile(lower)
    upper_bound = df[quant_var].quantile(upper)

    mask = df[quant_var].between(lower_bound, upper_bound)
    trimmed_df = df[mask]

    n_rows_before = len(df)
    n_rows_after = len(trimmed_df)
    n_rows_dropped = n_rows_before - n_rows_after

    summary = pd.Series({
        "quant_var": quant_var,
        "lower_quantile": lower,
        "upper_quantile": upper,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_rows_dropped": n_rows_dropped,
        "rows_dropped_share": n_rows_dropped / n_rows_before,
        "n_dropped_from_lower": (df[quant_var] < lower_bound).sum(),
        "n_dropped_from_upper": (df[quant_var] > upper_bound).sum(),
        "n_missing_in_var": df[quant_var].isna().sum(),
    })

    outputs = [trimmed_df[quant_var]]

    if return_trimmed_df:
        outputs.append(trimmed_df)
    if return_summary:
        outputs.append(summary)

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)



def cap_quantitative_var(
    df: pd.DataFrame,
    quant_var: str,
    cap_quantile: float = 0.90,
    return_capped_var: bool = True,
    return_capped_df: bool = False,
    drop_original_var: bool = False,
    return_summary: bool = False
) -> Union[
    pd.Series,
    pd.DataFrame,
    pd.Series,
    Tuple[pd.Series, pd.DataFrame],
    Tuple[pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.Series, pd.DataFrame, pd.Series]
]:
    """
    Cap values of a quantitative variable above a specified quantile.

    Values greater than the quantile threshold are replaced with the value
    of that quantile. Missing values are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    quant_var : str
        Quantitative variable to cap.
    cap_quantile : float, default=0.90
        Quantile used to define the upper cap. Must satisfy 0 < cap_quantile < 1.
    return_capped_var : bool, default=True
        Whether to return the capped variable as a Series.
    return_capped_df : bool, default=False
        Whether to return a dataframe with the capped variable added.
    drop_original_var : bool, default=False
        Whether to drop the original variable from the returned dataframe.
        Used only when return_capped_df=True.
    return_summary : bool, default=False
        Whether to return a summary of the capping operation.

    Returns
    -------
    pd.Series
        Capped variable.

    Optionally also returns
    -----------------------
    pd.DataFrame
        Dataframe with capped variable added.
    pd.Series
        Summary of the capping operation.
    """

    if df.shape[0] == 0:
        raise ValueError("Input dataframe has no rows.")
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")
    if quant_var not in df.columns:
        raise KeyError(f"Column '{quant_var}' not found.")
    if not 0 < cap_quantile < 1:
        raise ValueError("Require 0 < cap_quantile < 1.")
    
    df_out = df.copy()

    upper_bound = df[quant_var].quantile(cap_quantile)
    capped_col = f"{quant_var}_capped_{str(cap_quantile).replace('.', '_')}"

    df_out[capped_col] = df_out[quant_var].clip(upper=upper_bound)

    n_non_missing = df_out[quant_var].notna().sum()
    n_rows_capped = (df_out[quant_var] > upper_bound).sum()

    summary = pd.Series({
        "quant_var": quant_var,
        "capped_col": capped_col,
        "cap_quantile": cap_quantile,
        "upper_bound": upper_bound,
        "n_rows_total": len(df_out),
        "n_non_missing": int(n_non_missing),
        "n_missing_in_var": int(df_out[quant_var].isna().sum()),
        "n_rows_capped": int(n_rows_capped),
        "rows_capped_share_total": n_rows_capped / len(df_out),
        "rows_capped_share_non_missing": n_rows_capped / n_non_missing if n_non_missing > 0 else np.nan,
    })

    outputs = []

    if return_capped_var:
        outputs.append(df_out[capped_col])

    if return_capped_df:
        if drop_original_var:
            outputs.append(df_out.drop(columns=[quant_var]))
        else:
            outputs.append(df_out)

    if return_summary:
        outputs.append(summary)

    if len(outputs) == 0:
        return df_out[capped_col]

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)


def categorical_target_summary(
    df: pd.DataFrame,
    cat_var: str,
    target_var: str,
    include_missing: bool = True,
    sort_by_count: bool = True
) -> pd.DataFrame:
    """
    Return a summary table for a categorical variable with:
    - category counts
    - category share in dataset
    - TARGET value counts
    - TARGET value shares within each category

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    cat_var : str
        Categorical variable name.
    target_var : str
        Binary target variable name.
    include_missing : bool, default=True
        If True, missing values in cat_var are treated as a separate category 'Missing'.
        Rows with missing target are always dropped.
    sort_by_count : bool, default=True
        If True, sort result by category count descending.

    Returns
    -------
    pd.DataFrame
        Summary dataframe.
    """

    # Checks
    if cat_var not in df.columns:
        raise KeyError(f"Column '{cat_var}' not found.")
    if target_var not in df.columns:
        raise KeyError(f"Column '{target_var}' not found.")

    data = df[[cat_var, target_var]].copy()

    # Drop rows with missing target
    data = data[data[target_var].notna()].copy()

    if data.empty:
        raise ValueError("No rows with non-missing target.")

    # Handle missing categories
    if include_missing:
        data[cat_var] = data[cat_var].astype("object")
        data[cat_var] = data[cat_var].where(data[cat_var].notna(), "Missing")
    else:
        data = data[data[cat_var].notna()].copy()

    if data.empty:
        raise ValueError("No rows left after handling missing values.")

    # Category count and category share
    cat_count = data[cat_var].value_counts()
    cat_share = data[cat_var].value_counts(normalize=True)

    # Crosstab for target counts
    target_counts = pd.crosstab(data[cat_var], data[target_var])

    # Crosstab for target shares within category
    target_shares = pd.crosstab(data[cat_var], data[target_var], normalize="index")

    # Rename columns
    target_counts.columns = [f"target_{col}_count" for col in target_counts.columns]
    target_shares.columns = [f"target_{col}_share" for col in target_shares.columns]

    # Combine all together
    summary = pd.concat(
        [
            cat_count.rename("category_count"),
            cat_share.rename("category_share"),
            target_counts,
            target_shares
        ],
        axis=1
    ).reset_index()

    summary = summary.rename(columns={cat_var: "category"})

    if sort_by_count:
        summary = summary.sort_values("category_count", ascending=False).reset_index(drop=True)

    return summary