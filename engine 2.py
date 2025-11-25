# engine.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from functions import (markowitz_long_only,
                       load_price_panel,
                       load_composition_panel,
                       normalize_id,
                       load_metadata_panel,
                       load_esg_scores,
                       classify_esg,
                       filter_equity_candidates,
                       check_sector_constraints_feasibility,
                       check_esg_constraints_feasibility,
                       select_other_assets,
                       check_asset_class_constraints_feasibility,
                       management_fee_from_wealth)

@dataclass
class PortfolioConfig:
    # Dates / horizon
    today_date: pd.Timestamp
    investment_horizon_years: int = 1
    est_months: int = 12          # length of estimation window in months
    rebalancing: int = 12         # 12=yearly, 3=quarterly, 1=monthly

    # Risk aversion
    gamma: float = 2.0            # will later be computed from questionnaire

    # Initial invested wealth
    initial_wealth: float = 1_000_000.0

    # Universe choice
    universe_choice: str = "SP500"  # or "MSCI"

    # Equity filters
    keep_sectors: Optional[List[str]] = None    # e.g. ['Technology', 'Health Care']
    keep_esg: Optional[List[str]] = None        # e.g. ['M', 'H']

    # Other asset classes
    selected_asset_classes_other: Optional[List[str]] = None  # ['Commodities', 'Fixed Income', ...]
    keep_ids_by_class: Optional[Dict[str, Optional[List[str]]]] = None

    # Constraints
    max_weight_per_asset: float = 0.05
    sector_constraints: Optional[Dict[str, Dict[str, float]]] = field(default_factory=dict)
    esg_constraints: Optional[Dict[str, Dict[str, float]]] = field(default_factory=dict)
    asset_class_constraints: Optional[Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Transaction cost
    transaction_cost_bps: float = 50.0



def load_all_data():
    """
    Load all prices, returns, compositions, metadata and ESG labels.
    Returns a dict with a clear structure, so we don't re-load in Streamlit.
    """

    # ---- S&P 500 ----
    prices_sp500, returns_sp500 = load_price_panel("Prices.xlsx", "S&P500")
    sp500_composition = load_composition_panel("Composition.xlsx", "S&P500 Comp")
    metadata_sp500 = load_metadata_panel("metadata.xlsx", sheet_name="S&P500")
    esg_raw_sp500 = load_esg_scores("ESG Score.xlsx", "S&P500")
    esg_sp500 = classify_esg(esg_raw_sp500)

    # ---- MSCI ----
    prices_msci, returns_msci = load_price_panel("Prices.xlsx", "MSCI")
    msci_composition = load_composition_panel("Composition.xlsx", "MSCI Comp")
    metadata_msci = load_metadata_panel("metadata.xlsx", sheet_name="MSCI")
    esg_raw_msci = load_esg_scores("ESG Score.xlsx", "MSCI")
    esg_msci = classify_esg(esg_raw_msci)

    # ---- Other asset classes ----
    prices_other, returns_other = load_price_panel("Prices.xlsx", sheet_name="Other Class Assets")
    metadata_other = load_metadata_panel("metadata.xlsx", sheet_name="Other Class Assets")

    # ---- Benchmarks ----
    prices_bench, returns_bench = load_price_panel("Prices.xlsx", sheet_name="Benchmarks")

    data = {
        "prices": {
            "SP500": prices_sp500,
            "MSCI": prices_msci,
            "Other": prices_other,
            "Benchmarks": prices_bench,
        },
        "returns": {
            "SP500": returns_sp500,
            "MSCI": returns_msci,
            "Other": returns_other,
            "Benchmarks": returns_bench
        },
        "composition": {
            "SP500": sp500_composition,
            "MSCI": msci_composition,
        },
        "metadata": {
            "SP500": metadata_sp500,
            "MSCI": metadata_msci,
            "Other": metadata_other,
        },
        "esg_labels": {
            "SP500": esg_sp500,
            "MSCI": esg_msci,
        },
        "benchmarks": returns_bench,
    }

    return data


def run_backtest(config: PortfolioConfig, data: dict):
    """
    Run the full backtest given a configuration and pre-loaded data.

    Returns:
        perf : DataFrame with columns ['Rp', 'Growth'], index = Date
        summary_df : DataFrame with Top1/Top2/Top3/Top3_Total/Num_Assets per rebalance
        debug_weights_df : DataFrame with weights and metadata for each rebalance
    """

    # -------------------- Select equity universe --------------------
    if config.universe_choice == "SP500":
        returns_equity = data["returns"]["SP500"]
        composition_equity = data["composition"]["SP500"]
        metadata_equity = data["metadata"]["SP500"]
        esg_equity = data["esg_labels"]["SP500"]
    elif config.universe_choice == "MSCI":
        returns_equity = data["returns"]["MSCI"]
        composition_equity = data["composition"]["MSCI"]
        metadata_equity = data["metadata"]["MSCI"]
        esg_equity = data["esg_labels"]["MSCI"]
    else:
        raise ValueError(f"Unknown universe_choice: {config.universe_choice}")

    # Other asset classes
    returns_other = data["returns"]["Other"]
    metadata_other = data["metadata"]["Other"]

    # Combine equity + other asset classes
    returns_all = pd.concat([returns_equity, returns_other], axis=1).sort_index()
    metadata_all = pd.concat([metadata_equity, metadata_other], axis=0)

    # -------------------- Select other assets according to config --------------------
    other_ids_selected = select_other_assets(
        metadata_other=metadata_other,
        selected_asset_classes=config.selected_asset_classes_other,
        keep_ids_by_class=config.keep_ids_by_class,
    )

    # -------------------- Time grid for backtest --------------------
    portfolio_returns = []
    all_weights_summary = []
    debug_weights_rows = []
    prev_weights_end = None

    today_date = config.today_date
    investment_horizon_years = config.investment_horizon_years
    est_months = config.est_months
    rebalancing = config.rebalancing
    gamma = config.gamma
    mgmt_fee_annual = management_fee_from_wealth(config.initial_wealth)
    mgmt_fee_month = mgmt_fee_annual / 12.0

    # backtest start
    backtest_start_date = today_date - relativedelta(years=investment_horizon_years)
    start_month = backtest_start_date.to_period("M")

    # data bounds based on equity returns
    ret_min, ret_max = returns_equity.index.min(), returns_equity.index.max()

    # constraint 1: need est_months of history BEFORE each rebalance
    earliest_by_history = ret_min + est_months

    # constraint 2: need rebalancing-months of forward returns for the test window
    latest_by_future = ret_max - (rebalancing - 1)

    # final allowed interval for rebalances
    earliest_rebalance = max(start_month, earliest_by_history)
    latest_rebalance = latest_by_future

    if earliest_rebalance <= latest_rebalance:
        month_list = pd.period_range(earliest_rebalance, latest_rebalance, freq="M")
        rebalance_months = month_list[::rebalancing]
    else:
        rebalance_months = pd.PeriodIndex([], freq="M")

    # -------------------- Main rebalance loop --------------------
    for rebalance_month in rebalance_months:
        # 1) Date for candidates picking (index composition)
        candidates_period = rebalance_month - 1  # previous month for composition

        # 2) Estimation window dates (e.g. 12 months)
        estimation_end = candidates_period
        estimation_start = estimation_end - (est_months - 1)

        # 3) Test window dates
        test_start = rebalance_month
        test_end = test_start + (rebalancing - 1)

        # ---------- Build equity candidates ----------
        raw_candidates = (
            composition_equity[candidates_period]
            .dropna()
            .map(normalize_id)
            .dropna()
            .tolist()
        )

        filtered_candidates = filter_equity_candidates(
            raw_candidates=raw_candidates,
            candidates_period=candidates_period,
            metadata_equity=metadata_equity,
            esg_equity=esg_equity,
            keep_sectors=config.keep_sectors,
            keep_esg=config.keep_esg,
        )

        if len(filtered_candidates) == 0:
            # No candidates – skip this rebalance
            continue

        # Combine equity IDs + other asset IDs into the universe
        equity_ids = filtered_candidates
        universe_ids = pd.Index(equity_ids).append(other_ids_selected)
        universe_ids = pd.Index(sorted(set(universe_ids)))

        # Check that all IDs exist in returns_all
        missing = sorted(set(universe_ids) - set(returns_all.columns))
        if missing:
            raise ValueError(
                f"{len(missing)} universe IDs are missing in returns_all. "
                f"First few: {missing[:20]}"
            )

        # Estimation window of returns for ALL assets
        estimation_window = returns_all.loc[estimation_start:estimation_end, universe_ids]

        # Drop assets with any NaN over this estimation window
        bad = estimation_window.columns[estimation_window.isna().any()].tolist()
        if bad:
            # Optional: you can log or collect info here
            estimation_window = estimation_window.dropna(axis=1, how="any")

        # If everything got dropped, skip this rebalance
        if estimation_window.shape[1] == 0:
            continue

        # Sector / ESG / asset class vectors for ALL assets
        sector_for_assets = metadata_all["SECTOR"].reindex(estimation_window.columns)

        if candidates_period in esg_equity.index:
            esg_for_assets = esg_equity.loc[candidates_period].reindex(estimation_window.columns)
        else:
            esg_for_assets = pd.Series(index=estimation_window.columns, dtype=object)

        asset_class_for_assets = metadata_all["ASSET_CLASS"].reindex(estimation_window.columns)

        # Feasibility checks
        check_sector_constraints_feasibility(
            estimation_window.columns,
            metadata_all,
            config.sector_constraints,
        )
        check_esg_constraints_feasibility(
            esg_for_assets,
            config.esg_constraints,
        )
        check_asset_class_constraints_feasibility(
            estimation_window.columns,
            metadata_all,
            config.asset_class_constraints,
        )

        # ---------- Optimization ----------
        weights_t0 = markowitz_long_only(
            estimation_window,
            gamma=gamma,
            max_weight_per_asset=config.max_weight_per_asset,
            asset_class_for_assets=asset_class_for_assets,
            sector_for_assets=sector_for_assets,
            sector_constraints=config.sector_constraints,
            esg_for_assets=esg_for_assets,
            esg_constraints=config.esg_constraints,
            asset_class_constraints=config.asset_class_constraints,
        )

        weights_t0.name = str(rebalance_month)
        w_initial = weights_t0.values

        # TURNOVER & FEES AT THIS REBALANCE
        tc_rate = config.transaction_cost_bps / 10_000.0

        if prev_weights_end is None:
            # First rebalance: we start from cash (weights = 0) → trades = w_new
            prev_aligned = pd.Series(0.0, index=weights_t0.index)
        else:
            # Align previous end-of-period weights to current universe
            prev_aligned = prev_weights_end.reindex(weights_t0.index).fillna(0.0)

        # One-way turnover = 0.5 * sum |w_new - w_old|
        diff = weights_t0 - prev_aligned
        turnover_this_reb = 0.5 * diff.abs().sum()

        # Transaction cost as fraction of portfolio value
        tx_cost_this_reb = tc_rate * float(turnover_this_reb)

        top3_weights = np.sort(w_initial)[-3:][::-1]  # largest 3, descending
        top3_sum = top3_weights.sum()

        # ---------- Debug weights with metadata ----------
        weights_df = pd.DataFrame({
            "ID": weights_t0.index,
            "Weight": weights_t0.values,
        })

        # Drop tiny weights if desired
        weights_df = weights_df[weights_df["Weight"].abs() > 1e-6]

        meta_subset = metadata_all.reindex(weights_df["ID"])
        weights_df["NAME"] = meta_subset["NAME"].values if "NAME" in meta_subset.columns else np.nan
        weights_df["SECTOR"] = meta_subset["SECTOR"].values
        weights_df["ASSET_CLASS"] = meta_subset["ASSET_CLASS"].values

        if candidates_period in esg_equity.index:
            esg_row = esg_equity.loc[candidates_period]
            weights_df["ESG"] = esg_row.reindex(weights_df["ID"]).values
        else:
            weights_df["ESG"] = np.nan

        weights_df["Rebalance_Month"] = rebalance_month

        debug_weights_rows.append(weights_df)

        # ---------- Summary row ----------
        all_weights_summary.append({
            "Rebalance_Month": rebalance_month,
            "Top1": top3_weights[0],
            "Top2": top3_weights[1],
            "Top3": top3_weights[2],
            "Top3_Total": top3_sum,
            "Num_Assets": len(w_initial),
        })

        # ---------- Performance evaluation ----------
        test_window = returns_all.loc[test_start:test_end, weights_t0.index]
        rtw_sorted = test_window.sort_index()
        w_aligned = weights_t0.reindex(rtw_sorted.columns).fillna(0.0).astype(float)
        w = w_aligned / w_aligned.sum()
        rtw_adj = rtw_sorted.fillna(0.0)

        for k, (dt, r_vec) in enumerate(rtw_adj.iterrows()):
            r_vec = r_vec.astype(float)

            # Gross portfolio return for this month
            Rp_gross = float((w * r_vec).sum())

            # Transaction cost only at the first month after rebalance
            if k == 0:
                cost_frac = tx_cost_this_reb  # fraction of portfolio value
                turnover = float(turnover_this_reb)
            else:
                cost_frac = 0.0
                turnover = 0.0

            # 1) Net return after trading cost only: (1 - c)*(1+Rp_gross) - 1
            Rp_net_trading = (1.0 - cost_frac) * (1.0 + Rp_gross) - 1.0

            # 2) Apply MANAGEMENT fee (charged every month on total AUM)
            Rp_net = (1.0 - mgmt_fee_month) * (1.0 + Rp_net_trading) - 1.0

            portfolio_returns.append({
                "Date": dt,
                "Rp": Rp_net,          # net of trading + management fees
                "Rp_gross": Rp_gross,  # gross (no fees)
                "Turnover": turnover,  # non-zero only on rebalance month
                "TxCost": cost_frac,   # trading cost fraction
                "MgmtFee": mgmt_fee_month,  # management fee fraction this month
            })

            # Drift weights using GROSS returns (fees don't change weights, only wealth)
            w = w * (1.0 + r_vec)
            w = w / (1.0 + Rp_gross)

        # Save end-of-period weights for next turnover calculation
        prev_weights_end = w.copy()
        prev_weights_end.index = rtw_adj.columns

    # -------------------- Build outputs --------------------
    if portfolio_returns:
        perf = pd.DataFrame(portfolio_returns).set_index("Date").sort_index()

        # Net-of-fees growth
        perf["Growth"] = (1.0 + perf["Rp"]).cumprod()
        perf["CumReturn"] = perf["Growth"] - 1.0

        initial_wealth = getattr(config, "initial_wealth", 1_000_000.0)
        perf["Wealth"] = initial_wealth * perf["Growth"]

        # Gross growth
        perf["Growth_gross"] = (1.0 + perf["Rp_gross"]).cumprod()
        perf["CumReturn_gross"] = perf["Growth_gross"] - 1.0
    else:
        perf = pd.DataFrame(columns=["Rp", "Rp_gross", "Growth", "CumReturn", "Wealth",
                                     "Growth_gross", "CumReturn_gross",
                                     "Turnover", "TxCost"])

    if all_weights_summary:
        summary_df = pd.DataFrame(all_weights_summary)
        summary_df["Year"] = summary_df["Rebalance_Month"].dt.year
        summary_df = summary_df[
            ["Year", "Rebalance_Month", "Top1", "Top2", "Top3", "Top3_Total", "Num_Assets"]
        ]
    else:
        summary_df = pd.DataFrame(
            columns=["Year", "Rebalance_Month", "Top1", "Top2", "Top3", "Top3_Total", "Num_Assets"]
        )

    if debug_weights_rows:
        debug_weights_df = pd.concat(debug_weights_rows, ignore_index=True)
    else:
        debug_weights_df = pd.DataFrame(
            columns=["ID", "Weight", "NAME", "SECTOR", "ASSET_CLASS", "ESG", "Rebalance_Month"]
        )

    return perf, summary_df, debug_weights_df

# engine.py (continue)
from typing import Dict, Any
import pandas as pd
import numpy as np

from functions import (
    normalize_id,
    filter_equity_candidates,
    select_other_assets,
    check_sector_constraints_feasibility,
    check_esg_constraints_feasibility,
    check_asset_class_constraints_feasibility,
    markowitz_long_only,
)

def run_today_optimization(config: PortfolioConfig, data: dict) -> Dict[str, Any]:
    """
    One-shot optimization as of the latest month where we have both
    composition and returns.

    Returns a dict with:
        - 'candidates_period' : Period[M] used as "today"
        - 'weights' : DataFrame with ID, Weight, NAME, SECTOR, ASSET_CLASS, ESG
        - 'top5' : DataFrame of top 5 positions
        - 'alloc_by_asset_class' : Series
        - 'sector_in_equity' : Series (shares within equity slice)
        - 'esg_in_equity' : Series (shares within equity slice)
        - 'within_non_equity_classes' : dict[asset_class -> Series]
    """

    # -------------------- Select equity universe --------------------
    if config.universe_choice == "SP500":
        returns_equity = data["returns"]["SP500"]
        composition_equity = data["composition"]["SP500"]
        metadata_equity = data["metadata"]["SP500"]
        esg_equity = data["esg_labels"]["SP500"]
    elif config.universe_choice == "MSCI":
        returns_equity = data["returns"]["MSCI"]
        composition_equity = data["composition"]["MSCI"]
        metadata_equity = data["metadata"]["MSCI"]
        esg_equity = data["esg_labels"]["MSCI"]
    else:
        raise ValueError(f"Unknown universe_choice: {config.universe_choice}")

    # Other asset classes
    returns_other = data["returns"]["Other"]
    metadata_other = data["metadata"]["Other"]

    # Combine equity + other asset classes
    returns_all = pd.concat([returns_equity, returns_other], axis=1).sort_index()
    metadata_all = pd.concat([metadata_equity, metadata_other], axis=0)

    # -------------------- Select other assets --------------------
    other_ids_selected = select_other_assets(
        metadata_other=metadata_other,
        selected_asset_classes=config.selected_asset_classes_other,
        keep_ids_by_class=config.keep_ids_by_class,
    )

    # -------------------- Determine "today" month --------------------
    # Last month where we have both composition and returns
    comp_max = composition_equity.columns.max()      # Period[M]
    ret_max_all = returns_all.index.max()            # Period[M]
    candidates_period_today = min(comp_max, ret_max_all)

    est_months = config.est_months
    gamma = config.gamma

    estimation_end_today = candidates_period_today
    estimation_start_today = estimation_end_today - (est_months - 1)

    # -------------------- Equity candidates at "today" --------------------
    raw_candidates_today = (
        composition_equity[candidates_period_today]
        .dropna()
        .map(normalize_id)
        .dropna()
        .tolist()
    )

    filtered_candidates_today = filter_equity_candidates(
        raw_candidates=raw_candidates_today,
        candidates_period=candidates_period_today,
        metadata_equity=metadata_equity,
        esg_equity=esg_equity,
        keep_sectors=config.keep_sectors,
        keep_esg=config.keep_esg,
    )

    if len(filtered_candidates_today) == 0:
        raise ValueError(
            f"[Today optimization] No equity candidates left after filters at {candidates_period_today}."
        )

    # Combine equity IDs + other asset IDs into one universe
    equity_ids_today = filtered_candidates_today
    universe_ids_today = pd.Index(equity_ids_today).append(other_ids_selected)
    universe_ids_today = pd.Index(sorted(set(universe_ids_today)))

    missing_today = sorted(set(universe_ids_today) - set(returns_all.columns))
    if missing_today:
        raise ValueError(
            f"[Today optimization] {len(missing_today)} universe IDs are missing in returns_all. "
            f"First few: {missing_today[:20]}"
        )

    # Estimation window of returns for ALL assets (today)
    estimation_window_today = returns_all.loc[estimation_start_today:estimation_end_today, universe_ids_today]

    bad_today = estimation_window_today.columns[estimation_window_today.isna().any()].tolist()
    if bad_today:
        # Optional logging; we just drop them
        estimation_window_today = estimation_window_today.dropna(axis=1, how="any")

    if estimation_window_today.shape[1] == 0:
        raise ValueError("[Today optimization] All assets dropped due to NaNs in estimation window.")

    # -------------------- Metadata vectors --------------------
    sector_for_assets_today = metadata_all["SECTOR"].reindex(estimation_window_today.columns)

    if candidates_period_today in esg_equity.index:
        esg_for_assets_today = esg_equity.loc[candidates_period_today].reindex(estimation_window_today.columns)
    else:
        esg_for_assets_today = pd.Series(index=estimation_window_today.columns, dtype=object)

    asset_class_for_assets_today = metadata_all["ASSET_CLASS"].reindex(estimation_window_today.columns)

    # -------------------- Feasibility checks --------------------
    check_sector_constraints_feasibility(
        estimation_window_today.columns,
        metadata_all,
        config.sector_constraints,
    )
    check_esg_constraints_feasibility(
        esg_for_assets_today,
        config.esg_constraints,
    )
    check_asset_class_constraints_feasibility(
        estimation_window_today.columns,
        metadata_all,
        config.asset_class_constraints,
    )

    # -------------------- Optimize --------------------
    weights_today = markowitz_long_only(
        estimation_window_today,
        gamma=gamma,
        max_weight_per_asset=config.max_weight_per_asset,
        asset_class_for_assets=asset_class_for_assets_today,
        sector_for_assets=sector_for_assets_today,
        sector_constraints=config.sector_constraints,
        esg_for_assets=esg_for_assets_today,
        esg_constraints=config.esg_constraints,
        asset_class_constraints=config.asset_class_constraints,
    )

    weights_today.name = f"Today_{candidates_period_today}"

    # -------------------- Build detailed output DataFrame --------------------
    today_df = pd.DataFrame({
        "ID": weights_today.index,
        "Weight": weights_today.values,
    })

    # Drop tiny weights
    today_df = today_df[today_df["Weight"].abs() > 1e-6]

    meta_today = metadata_all.reindex(today_df["ID"])
    today_df["NAME"] = meta_today["NAME"].values if "NAME" in meta_today.columns else np.nan
    today_df["SECTOR"] = meta_today["SECTOR"].values
    today_df["ASSET_CLASS"] = meta_today["ASSET_CLASS"].values

    if candidates_period_today in esg_equity.index:
        esg_row_today = esg_equity.loc[candidates_period_today]
        today_df["ESG"] = esg_row_today.reindex(today_df["ID"]).values
    else:
        today_df["ESG"] = np.nan

    # -------------------- Summaries --------------------
    # 1) Top 5 positions
    top5_today = today_df.sort_values("Weight", ascending=False).head(5)

    # 2) Allocation by asset class (total portfolio)
    alloc_by_ac_today = (
        today_df.groupby("ASSET_CLASS")["Weight"].sum().sort_values(ascending=False)
    )

    # 3) Within equity: sector & ESG as % of equity slice
    equity_slice_today = today_df[today_df["ASSET_CLASS"] == "Equity"].copy()
    total_equity_weight_today = equity_slice_today["Weight"].sum()

    if total_equity_weight_today > 0:
        sector_in_equity_today = (
            equity_slice_today.groupby("SECTOR")["Weight"].sum() / total_equity_weight_today
        ).sort_values(ascending=False)

        esg_in_equity_today = (
            equity_slice_today.groupby("ESG")["Weight"].sum() / total_equity_weight_today
        ).sort_values(ascending=False)
    else:
        sector_in_equity_today = pd.Series(dtype=float)
        esg_in_equity_today = pd.Series(dtype=float)

    # 4) Within each non-equity asset class: allocation by asset (as % of that class)
    non_equity_classes_today = today_df["ASSET_CLASS"].dropna().unique().tolist()
    non_equity_classes_today = [ac for ac in non_equity_classes_today if ac != "Equity"]

    within_class_allocations_today: Dict[str, pd.Series] = {}
    for ac in non_equity_classes_today:
        slice_ac = today_df[today_df["ASSET_CLASS"] == ac].copy()
        total_ac_weight = slice_ac["Weight"].sum()
        if total_ac_weight > 0:
            within_class_allocations_today[ac] = (
                slice_ac.set_index("NAME")["Weight"] / total_ac_weight
            ).sort_values(ascending=False)

    return {
        "candidates_period": candidates_period_today,
        "weights": today_df,
        "top5": top5_today,
        "alloc_by_asset_class": alloc_by_ac_today,
        "sector_in_equity": sector_in_equity_today,
        "esg_in_equity": esg_in_equity_today,
        "within_non_equity_classes": within_class_allocations_today,
    }


