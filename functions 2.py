# functions.py
import pandas as pd
import numpy as np
import os
import pickle
import itertools
from datetime import datetime
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf


def normalize_id(x):
    """
    function to normalize every cell to consistent ID format
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # strip trailing ".0" if it came from Excel as a float-looking code
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()


def to_month_period(c):
    """
    convert columns to month periods; leave any non-date columns untouched if present
    """

    try:
        return pd.to_datetime(c).to_period('M')
    except Exception:
        return c


def load_price_panel(excel_path, sheet_name=None):
    """
    Reads a monthly price file from a given sheet in an Excel workbook.
    First column = ID, others = dates.

    Returns:
        prices: DataFrame (index = Date as Period[M], columns = asset IDs)
        returns: DataFrame (same shape, monthly returns)
    """

    df = pd.read_excel(excel_path,sheet_name=sheet_name)

    # first column is ID
    df = df.rename(columns={df.columns[0]: 'id'})
    df['id'] = df['id'].map(normalize_id)
    df = df.set_index('id')

    # columns -> monthly PeriodIndex
    df.columns = pd.to_datetime(df.columns).to_period('M')

    # prices with Date as index
    prices = df.T
    prices.index.name = 'Date'

    # compute returns: r_t = P_{t+1}/P_t - 1
    returns = prices.shift(-1).divide(prices) - 1
    returns = returns.iloc[:-1]

    return prices, returns


def load_composition_panel(excel_path, sheet_name=None):
    """
    Reads a composition file (or sheet) with columns = months, rows = assets.
    Each column lists the tickers held in that month.

    excel_path : path to Excel file
    sheet_name : sheet name inside the workbook (None = first sheet)
    """

    comp = pd.read_excel(excel_path, sheet_name=sheet_name)

    # convert columns to Period[M]
    comp.columns = [to_month_period(c) for c in comp.columns]
    comp.columns = pd.PeriodIndex(comp.columns, freq='M')

    # normalize IDs in every column
    for c in comp.columns:
        comp[c] = comp[c].map(normalize_id)

    return comp


def load_metadata_panel(excel_path, sheet_name=None):
    """
    Load metadata (ID, NAME, ISIN, TICKER, SECTOR) for one universe.

    excel_path : path to metadata Excel file
    sheet_name : sheet name ('S&P500', 'MSCI', etc.)

    Returns:
        DataFrame indexed by normalized ID with columns:
        ['NAME', 'ISIN', 'TICKER', 'SECTOR', ...]
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Assume first column is 'Type' = internal ID
    if 'Type' in df.columns:
        df = df.rename(columns={'Type': 'id'})
    else:
        df = df.rename(columns={df.columns[0]: 'id'})

    # Normalize IDs
    df['id'] = df['id'].map(normalize_id)

    # Set ID as index
    df = df.set_index('id')

    # Standardize ASSET CLASS column name if present
    # e.g. "ASSET CLASS" -> "ASSET_CLASS"
    if 'ASSET CLASS' in df.columns:
        df = df.rename(columns={'ASSET CLASS': 'ASSET_CLASS'})

    # Ensure SECTOR column exists (for equities); if missing, fill with NaN
    if 'SECTOR' in df.columns:
        df['SECTOR'] = df['SECTOR'].astype(str).str.strip()
    else:
        df['SECTOR'] = np.nan

    return df


def load_esg_scores(excel_path, sheet_name=None):
    """
    Load ESG panel where rows = dates, columns = company IDs, values = ESG numeric score.
    Converts index to Period[M] and normalizes tickers.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M')
    df = df.rename(columns={date_col: 'Date'})
    df = df.set_index('Date')
    df.columns = [normalize_id(c) for c in df.columns]

    return df

def classify_esg(df):
    """
    Input:
        df: DataFrame indexed by Period[M], columns = asset IDs,
            values = numeric ESG scores.
    Output:
        DataFrame indexed by Period[M], same columns, values = 'L','M','H'
    """

    def classify_row(row):
        # row: ESG scores for one Date, index = asset IDs
        s = row.dropna()

        if s.empty:
            # no data for this date
            return pd.Series(index=row.index, dtype=object)

        # quantiles on available scores
        Q1 = np.nanpercentile(s, 25)
        Q3 = np.nanpercentile(s, 75)

        # labels defined only on non-NaN assets
        labels = pd.Series(index=s.index, dtype=object)
        labels[s < Q1] = "L"
        labels[(s >= Q1) & (s < Q3)] = "M"
        labels[s >= Q3] = "H"

        # reindex back to the full row index (all assets),
        # missing ones become NaN labels
        return labels.reindex(row.index)

    labels_df = df.apply(classify_row, axis=1)
    return labels_df


def filter_equity_candidates(raw_candidates,
                             candidates_period,
                             metadata_equity,
                             esg_equity,
                             keep_sectors=None,
                             keep_esg=None):
    """
    raw_candidates : list of IDs from composition (already normalized ideally)
    candidates_period : Period[M] of the rebalance/candidate date
    metadata_equity : DataFrame indexed by ID, with at least column 'SECTOR'
    esg_equity : DataFrame indexed by Period[M], columns = IDs, values = 'L','M','H'
    keep_sectors : list of sector names to KEEP (None = keep all)
    keep_esg : list of ESG labels to KEEP, e.g. ['M','H'] (None = keep all)

    Returns:
        filtered_candidates : list of IDs that pass all filters
    """

    # Start from a clean Index of IDs
    ids = pd.Index(raw_candidates).dropna()

    # ---------- Sector filter ----------
    if keep_sectors is not None and len(keep_sectors) > 0:
        # Get sectors for these IDs
        sectors = metadata_equity.reindex(ids)['SECTOR']
        mask = sectors.isin(keep_sectors)

        filtered_ids = sectors.index[mask]
        # (optional) debug prints:
        # dropped = sectors.index[~mask | sectors.isna()]
        # print(f"Dropped {len(dropped)} by sector filter.")
    else:
        filtered_ids = ids

    # ---------- ESG filter ----------
    if keep_esg is not None and len(keep_esg) > 0:
        if candidates_period in esg_equity.index:
            esg_row = esg_equity.loc[candidates_period]
            esg_for_ids = esg_row.reindex(filtered_ids)

            mask_esg = esg_for_ids.isin(keep_esg)
            filtered_ids = esg_for_ids.index[mask_esg]
            # (optional) debug:
            # dropped_esg = esg_for_ids.index[~mask_esg | esg_for_ids.isna()]
            # print(f"Dropped {len(dropped_esg)} by ESG filter.")
        else:
            # No ESG data for this month → skip ESG filter
            # print(f"No ESG data for {candidates_period}, skipping ESG filter.")
            pass

    return list(filtered_ids)


def markowitz_long_only(estimation_window,
                        gamma=None,
                        max_weight_per_asset=0.05,
                        asset_class_for_assets=None,
                        sector_for_assets=None,
                        sector_constraints=None,
                        esg_for_assets=None,
                        esg_constraints=None,
                        asset_class_constraints=None):
    """
    estimation_window : DataFrame of returns, columns = assets, rows = months
    gamma : risk aversion parameter (must be > 0)
    max_weight_per_asset : upper bound per asset (e.g. 0.10 for 10%)
    asset_class_for_assets : pd.Series indexed by asset ID, giving asset class
    sector_for_assets : pd.Series indexed by asset ID, giving sector name
    sector_constraints : dict, e.g.
        {
            'Information Technology': {'max': 0.20},
            'Health Care': {'min': 0.10},
        }
    esg_for_assets : pd.Series indexed by asset ID, giving 'L','M','H' or NaN
    esg_constraints : dict, same style as sector_constraints
    asset_class_constraints : dict, e.g.
        {
            'Equity': {'min': 0.7},
            'Fixed Income': {'max': 0.2},
        }

    This version uses cvxpy instead of scipy.optimize.minimize.
    """

    # ------------------ Basic sanity checks ------------------
    if estimation_window is None or estimation_window.shape[1] == 0:
        raise ValueError("markowitz_long_only: estimation_window has no assets (0 columns).")

    if gamma is None or gamma <= 0:
        raise ValueError(f"markowitz_long_only: gamma must be positive, got {gamma}.")

    assets = list(estimation_window.columns)
    n = len(assets)

    if max_weight_per_asset <= 0 or max_weight_per_asset > 1:
        raise ValueError(
            f"markowitz_long_only: max_weight_per_asset should be in (0,1], got {max_weight_per_asset}."
        )

    # ------------------ Estimate mu and Sigma ------------------
    mu_hat = estimation_window.mean(axis=0).values.astype(float)

    X = estimation_window.values  # rows=months, cols=assets
    lw = LedoitWolf().fit(X)
    sigma_hat = lw.covariance_.astype(float)

    # Safety: enforce positive definiteness by clipping eigenvalues
    eigvals, eigvecs = np.linalg.eigh(sigma_hat)
    eps = 1e-8
    eigvals_clipped = np.clip(eigvals, eps, None)
    sigma_hat = (eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T).astype(float)

    # ------------------ Equity mask (for relative constraints) ------------------
    if asset_class_for_assets is not None:
        asset_class_for_assets = asset_class_for_assets.reindex(assets)
        equity_mask = (asset_class_for_assets == "Equity").astype(float).values
    else:
        # fallback: treat all assets as "equity"
        equity_mask = np.ones(n, dtype=float)

    if (sector_constraints or esg_constraints) and equity_mask.sum() == 0:
        raise ValueError(
            "Sector/ESG constraints specified but no asset is labeled as 'Equity' "
            "in asset_class_for_assets."
        )

    # ------------------ cvxpy variable ------------------
    w = cp.Variable(n)

    constraints = []

    # Long-only, fully invested
    constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)

    # Per-asset upper bound
    constraints.append(w <= max_weight_per_asset)

    # ------------------ Sector constraints (relative to equity) ------------------
    if (sector_for_assets is not None) and (sector_constraints is not None):
        sector_for_assets = sector_for_assets.reindex(assets)

        for sector_name, cons in sector_constraints.items():
            if cons is None:
                continue

            mask = (sector_for_assets == sector_name).astype(float).values
            if mask.sum() == 0:
                continue

            sector_weight = mask @ w
            equity_weight = equity_mask @ w

            # Max: sector_weight / equity_weight <= cap
            #  => cap * equity_weight - sector_weight >= 0   (linear in w)
            if "max" in cons and cons["max"] is not None:
                cap = float(cons["max"])
                constraints.append(cap * equity_weight - sector_weight >= 0)

            # Min: sector_weight / equity_weight >= floor
            #  => sector_weight - floor * equity_weight >= 0
            if "min" in cons and cons["min"] is not None:
                floor = float(cons["min"])
                constraints.append(sector_weight - floor * equity_weight >= 0)

    # ------------------ ESG constraints (relative to equity) ------------------
    if (esg_for_assets is not None) and (esg_constraints is not None):
        esg_for_assets = esg_for_assets.reindex(assets)

        for label, cons in esg_constraints.items():
            if cons is None:
                continue

            mask = (esg_for_assets == label).astype(float).values
            if mask.sum() == 0:
                continue

            esg_weight = mask @ w
            equity_weight = equity_mask @ w

            # Max: esg_weight / equity_weight <= cap
            if "max" in cons and cons["max"] is not None:
                cap = float(cons["max"])
                constraints.append(cap * equity_weight - esg_weight >= 0)

            # Min: esg_weight / equity_weight >= floor
            if "min" in cons and cons["min"] is not None:
                floor = float(cons["min"])
                constraints.append(esg_weight - floor * equity_weight >= 0)

    # ------------------ Asset-class constraints (absolute on total portfolio) ------------------
    if (asset_class_for_assets is not None) and (asset_class_constraints is not None):
        asset_class_for_assets = asset_class_for_assets.reindex(assets)

        for ac_name, cons in asset_class_constraints.items():
            if cons is None:
                continue

            mask = (asset_class_for_assets == ac_name).astype(float).values
            if mask.sum() == 0:
                continue

            ac_weight = mask @ w

            # Max: sum w_i in class <= cap
            if "max" in cons and cons["max"] is not None:
                cap = float(cons["max"])
                constraints.append(ac_weight <= cap)

            # Min: sum w_i in class >= floor
            if "min" in cons and cons["min"] is not None:
                floor = float(cons["min"])
                constraints.append(ac_weight >= floor)

    # ------------------ Objective: mean–variance trade-off ------------------
    # Minimize: 0.5 * w'Σw − gamma * μ'w
    quad_term = 0.5 * cp.quad_form(w, sigma_hat)
    linear_term = -gamma * (mu_hat @ w)
    objective = cp.Minimize(quad_term + linear_term)

    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.OSQP,      # you can try ECOS or SCS as well
        warm_start=True,
        verbose=False,
    )

    if w.value is None:
        raise ValueError("Optimization failed in cvxpy: no solution returned.")

    w_opt = np.array(w.value).ravel().astype(float)

    # Numerical cleanup: clip tiny negatives to 0, renormalize
    w_opt = np.where(w_opt < 0, 0.0, w_opt)
    s = w_opt.sum()
    if s <= 0:
        raise ValueError("Optimization returned non-positive total weight.")
    w_opt /= s

    return pd.Series(w_opt, index=assets, name="weights_opt")


def check_sector_constraints_feasibility(assets, metadata_equity, sector_constraints):
    if sector_constraints is None:
        return

    # mapping: asset -> sector
    sector_for_assets = metadata_equity['SECTOR'].reindex(assets)

    # 1) total min cannot exceed 1
    total_min = sum(cons.get('min', 0) for cons in sector_constraints.values())
    if total_min > 1.0:
        raise ValueError(f"Total minimum sector weights {total_min:.2f} exceed 1.0")

    # 2) each sector with a min must appear in assets
    for sector_name, cons in sector_constraints.items():
        if 'min' in cons:
            if not any(sector_for_assets == sector_name):
                raise ValueError(
                    f"Sector '{sector_name}' has a min constraint but does not appear in the universe!"
                )

    # 3) consistency: min <= max
    for sector_name, cons in sector_constraints.items():
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"In sector '{sector_name}', min={cons['min']} > max={cons['max']}"
                )


def check_esg_constraints_feasibility(esg_for_assets, esg_constraints):
    if esg_constraints is None:
        return

    # 1) total min cannot exceed 1
    total_min = sum(cons.get('min', 0) for cons in esg_constraints.values())
    if total_min > 1.0:
        raise ValueError(f"Total minimum ESG weights {total_min:.2f} exceed 1.0")

    # 2) each ESG label with a min must exist in current assets
    for label, cons in esg_constraints.items():
        if 'min' in cons:
            if not any(esg_for_assets == label):
                raise ValueError(
                    f"ESG label '{label}' has a min constraint but no asset has this label in this universe/period!"
                )

    # 3) consistency: min <= max (if both present)
    for label, cons in esg_constraints.items():
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"For ESG '{label}', min={cons['min']} > max={cons['max']}"
                )


def select_other_assets(metadata_other,
                        selected_asset_classes=None,
                        keep_ids_by_class=None):
    """
    Select non-equity assets (Other Class) based on client choices.

    metadata_other : DataFrame indexed by ID, with column 'ASSET_CLASS'

    selected_asset_classes :
        - None  -> include all asset classes
        - []    -> include *no* other asset classes
        - list  -> include only those classes

    keep_ids_by_class : optional dict mapping asset class -> list of IDs to KEEP
        Example:
            {
                'Alternative Instruments': ['BTC', 'ETH'],
                'Commodities': None,
            }

    Returns:
        Index of IDs to include from metadata_other.
    """

    df = metadata_other.copy()

    # 1) Handle asset-class level selection
    if selected_asset_classes is None:
        # No filter -> keep all classes
        pass
    else:
        # If an empty list is explicitly passed -> no other asset classes
        if len(selected_asset_classes) == 0:
            return pd.Index([], dtype=df.index.dtype)

        # Otherwise keep only the requested classes
        df = df[df["ASSET_CLASS"].isin(selected_asset_classes)]

    # 2) If no per-class ID filter is provided -> keep all IDs of remaining classes
    if keep_ids_by_class is None:
        return df.index

    # 3) Apply per-class ID filters
    selected_ids = []

    for asset_class in df["ASSET_CLASS"].dropna().unique():
        df_class = df[df["ASSET_CLASS"] == asset_class]
        ids_class = df_class.index

        ids_to_keep = keep_ids_by_class.get(asset_class, None)
        if ids_to_keep is not None:
            ids_class = ids_class.intersection(pd.Index(ids_to_keep))

        selected_ids.extend(list(ids_class))

    return pd.Index(selected_ids)


def check_asset_class_constraints_feasibility(assets, metadata_all, asset_class_constraints):
    if asset_class_constraints is None:
        return

    asset_class_for_assets = metadata_all['ASSET_CLASS'].reindex(assets)

    # 1) total min cannot exceed 1
    total_min = 0.0
    for cons in asset_class_constraints.values():
        if cons is None:
            continue
        total_min += cons.get('min', 0)

    if total_min > 1.0:
        raise ValueError(f"Total minimum asset-class weights {total_min:.2f} exceed 1.0")

    # 2) each asset class with a min must appear in assets
    for ac_name, cons in asset_class_constraints.items():
        if cons is None:
            continue
        if 'min' in cons:
            if not any(asset_class_for_assets == ac_name):
                raise ValueError(
                    f"Asset class '{ac_name}' has a min constraint but does not appear in the universe!"
                )

    # 3) consistency: min <= max
    for ac_name, cons in asset_class_constraints.items():
        if cons is None:
            continue
        if 'min' in cons and 'max' in cons:
            if cons['min'] > cons['max']:
                raise ValueError(
                    f"In asset class '{ac_name}', min={cons['min']} > max={cons['max']}"
                )


def validate_constraints(
    sector_constraints: dict | None,
    esg_constraints: dict | None,
    asset_class_constraints: dict | None,
):
    """
    Basic feasibility checks for constraints after user input.

    - For each group (sector / ESG / asset class):
        - sum of mins <= 1
        - min <= max for each item (if both exist)
    """
    errors: list[str] = []

    # ----- Sector constraints -----
    if sector_constraints:
        total_min = 0.0
        for name, cons in sector_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"Sector '{name}': minimum share ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum sector shares** ({total_min:.2f}) exceeds 100% of the equity slice."
            )

    # ----- ESG constraints -----
    if esg_constraints:
        total_min = 0.0
        for label, cons in esg_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"ESG '{label} score': minimum share ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum ESG shares** ({total_min:.2f}) exceeds 100% of the equity slice."
            )

    # ----- Asset-class constraints -----
    if asset_class_constraints:
        total_min = 0.0
        for ac_name, cons in asset_class_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"Asset class '{ac_name}': minimum weight ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum asset-class weights** ({total_min:.2f}) exceeds 100% of the portfolio."
            )

    return errors


def compute_backtest_stats(perf: pd.DataFrame) -> dict:
    """
    Compute key performance statistics from the backtest result `perf`.

    perf: DataFrame with at least column 'Rp' (monthly returns).
          Index should be PeriodIndex (monthly) or DatetimeIndex.
    """

    if perf.empty or "Rp" not in perf.columns:
        return {}

    r = perf["Rp"].dropna().copy()

    if r.empty:
        return {}

    # --- Annualised average return (arithmetic) ---
    avg_return_m = r.mean()
    avg_return_y = avg_return_m * 12.0

    # --- Annualised volatility ---
    vol_m = r.std(ddof=1)
    vol_y = vol_m * np.sqrt(12.0)

    # --- Annualised cumulative return (geometric) ---
    T = len(r)
    total_return = (1.0 + r).prod() - 1.0
    annualised_cum_return = (1.0 + total_return) ** (12.0 / T) - 1.0

    # --- Min / max monthly return ---
    min_return = r.min()
    max_return = r.max()

    # --- Max drawdown, start, end, duration ---
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = (running_max - wealth) / running_max  # 0 … +X%

    max_dd = drawdown.max()

    if pd.isna(max_dd) or max_dd == 0:
        # no drawdown
        dd_end = dd_start = r.index[0]
        dd_duration = 0
    else:
        dd_end = drawdown.idxmax()                      # period of worst drawdown
        dd_start = wealth.loc[:dd_end].idxmax()         # previous peak
        # duration in months = number of periods between start and end
        dd_duration = len(wealth.loc[dd_start:dd_end]) - 1

    # Convert dates to Timestamp for display (nice in Streamlit)
    def to_ts(idx):
        if isinstance(idx, pd.Period):
            return idx.to_timestamp()
        return pd.to_datetime(idx)



    results = {
        "annualised_avg_return": avg_return_y,
        "annualised_volatility": vol_y,
        "annualised_cum_return": annualised_cum_return,
        "min_monthly_return": min_return,
        "max_monthly_return": max_return,
        "max_drawdown": max_dd,
        "max_drawdown_start": to_ts(dd_start),
        "max_drawdown_end": to_ts(dd_end),
        "max_drawdown_duration_months": int(dd_duration),
    }

    return results

def management_fee_from_wealth(initial_wealth: float) -> float:
    """
    Annual management fee as a decimal (e.g. 0.005 = 0.5% p.a.)
    based on initial invested wealth.

    - 1m – 10m : 0.50% per year
    - 10m – 20m: 0.45% per year
    - 20m – 30m: 0.40% per year
    - 30m – 50m: 0.35% per year
    - 50m - 100m: 0.30% per year
    - >100: 0.25% per year

    """

    w = float(initial_wealth)

    if w < 1_000_000:
        # below min ticket, still charge 0.50% for simplicity
        return 0.005

    if w < 10_000_000:
        return 0.005  # 0.50%
    elif w < 20_000_000:
        return 0.0045  # 0.45%
    elif w < 30_000_000:
        return 0.0040  # 0.40%
    elif w < 50_000_000:
        return 0.0035  # 0.35%
    elif w < 100_000_000:
        return 0.003  # 0.30%
    else:
        return 0.0025  # 0.25%


def build_backtest_context_text(
    stats,
    perf,
    investment_amount,
    universe_choice,
    investment_horizon_years,
    est_months,
    rebalancing,
    gamma,
    profile_label,
    max_weight_per_asset,
    selected_asset_classes_other,
    sector_constraints,
    esg_constraints,
    asset_class_constraints,
):
    """
    Build a concise text summary of the backtest and the client configuration
    to feed into the LLM for commentary.
    """
    start_date = perf.index[0]
    end_date = perf.index[-1]

    initial_wealth = investment_amount
    final_wealth = investment_amount * float(perf["Growth"].iloc[-1])

    # Human-readable labels
    universe_label = "S&P 500" if universe_choice == "SP500" else "MSCI World"

    if rebalancing == 12:
        rebalance_label = "Yearly"
    elif rebalancing == 3:
        rebalance_label = "Quarterly"
    else:
        rebalance_label = "Monthly"

    # Asset classes beyond equity
    other_ac = ", ".join(selected_asset_classes_other) if selected_asset_classes_other else "None (Equity-only)"

    # Simple summaries of constraints
    n_sector_cons = len(sector_constraints) if sector_constraints else 0
    n_esg_cons = len(esg_constraints) if esg_constraints else 0
    n_ac_cons = len(asset_class_constraints) if asset_class_constraints else 0

    # 🔹 Build human-readable summaries of constraints
    def fmt_bounds(name, cons_dict):
        parts = []
        for key, bounds in cons_dict.items():
            text = f"{key}:"
            if "min" in bounds:
                text += f" min {bounds['min']:.0%}"
            if "max" in bounds:
                text += f" max {bounds['max']:.0%}"
            parts.append(text)
        return "; ".join(parts) if parts else "None"

    sector_text = fmt_bounds("Sector", sector_constraints or {})
    esg_text = fmt_bounds("ESG", esg_constraints or {})
    ac_text = fmt_bounds("Asset class", asset_class_constraints or {})

    context = f"""
    Client configuration and backtest summary for Phi Investment Capital:

    Client inputs:
    - Equity universe: {universe_label}
    - Investment horizon: {investment_horizon_years} years
    - Estimation window: {est_months} months
    - Rebalancing frequency: {rebalance_label}
    - Initial invested wealth: {initial_wealth:,.0f}
    - Risk profile: {profile_label} (internal risk-aversion parameter gamma = {gamma:.2f})
    - Maximum weight per individual asset: {max_weight_per_asset:.0%}
    - Other asset classes selected in the universe (beyond equity): {other_ac}

    Constraints (qualitative view):
    - Number of active sector constraints (within equity): {n_sector_cons}
    - Number of active ESG constraints (within equity): {n_esg_cons}
    - Number of active asset-class constraints (total portfolio): {n_ac_cons}

    Detailed constraint bounds:
    - Sector constraints: {sector_text}
    - ESG constraints: {esg_text}
    - Asset-class constraints: {ac_text}

    Backtest period and performance:
    - Backtest period: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}
    - Final wealth at end of backtest: {final_wealth:,.0f}

    Key performance statistics:
    - Annualised average return: {stats['annualised_avg_return']:.2%}
    - Annualised volatility: {stats['annualised_volatility']:.2%}
    - Annualised cumulative return: {stats['annualised_cum_return']:.2%}
    - Minimum monthly return: {stats['min_monthly_return']:.2%}
    - Maximum monthly return: {stats['max_monthly_return']:.2%}
    - Maximum drawdown: {stats['max_drawdown']:.2%}
    - Max drawdown period: {stats['max_drawdown_start'].strftime('%b %Y')} to {stats['max_drawdown_end'].strftime('%b %Y')}
    - Max drawdown duration: {stats['max_drawdown_duration_months']} months
        """.strip()

    return context
