import operator
import numpy as np
import pandas as pd


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

            df_out = df_out.drop(columns="_Filla_value")

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