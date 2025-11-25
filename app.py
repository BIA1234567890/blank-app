# app.py
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq

from engine import (
    PortfolioConfig,
    load_all_data,
    run_backtest,
    run_today_optimization,
)

from functions import (validate_constraints,
                       compute_backtest_stats,
                       management_fee_from_wealth,
                       build_backtest_context_text)


# --------------- GLOBAL DATA (cached) ---------------
@st.cache_data
def get_data():
    return load_all_data()


def main():
    st.set_page_config(
        page_title="QARM Portfolio Manager",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("About us", "Portfolio optimization", "Phi assistant"),
    )

    data = get_data()

    if page == "About us":
        page_about()
    elif page == "Portfolio optimization":
        page_portfolio_optimization(data)
    elif page == "Phi assistant":
        page_ai_assistant()


def get_llm_client():
    """
    Returns a Groq client using the secret API key.
    """
    api_key = st.secrets["groq"]["api_key"]
    client = Groq(api_key=api_key)
    return client


# --------------- PAGE 1: ABOUT US ---------------
def page_about():
    st.title("Our Investment Firm")

    st.markdown(
        """
        ### Who we are
        We are a quantitative asset & risk management boutique.

        Our mission is to build **transparent, rule-based portfolios**
        tailored to each client's risk profile, constraints and ESG preferences.

        ### What this app does
        - Builds a diversified multi-asset portfolio (Equity, Fixed Income, Commodities, Alternatives)
        - Applies **sector** and **ESG** constraints inside the equity bucket
        - Applies **asset-class** constraints at the total portfolio level
        - Optimizes using a **Markowitz mean–variance** model with a robust covariance estimator (Ledoit–Wolf)
        - Backtests the strategy over the selected horizon

        Use the *Portfolio optimization* page from the sidebar to try it.
        """
    )


# --------------- PAGE 2: PORTFOLIO OPTIMIZATION ---------------
def page_portfolio_optimization(data):
    st.title("Portfolio Optimization")

    st.markdown(
        """
        This tool builds a **constrained multi-asset portfolio** based on your preferences:

        1. Choose the **market universe & technical settings**  
        2. Refine the **investment universe with filters** (sectors, ESG, asset classes)  
        3. Answer a short **risk profile questionnaire**  
        4. Set **portfolio constraints** (sectors, ESG, asset classes, max weights)  
        5. Run the **optimization & backtest** and analyze the results  
        """
    )

    # ============================================================
    # STEP 1 – GENERAL SETTINGS
    # ============================================================
    st.markdown("### Step 1 – General Settings")

    # ---- 4 columns for main settings ----
    colA, colB, colC, colD = st.columns(4)

    with colA:
        universe_choice = st.radio(
            "Equity Universe",
            options=["SP500", "MSCI"],
            format_func=lambda x: "S&P 500" if x == "SP500" else "MSCI World",
        )

    with colB:
        investment_amount = st.number_input(
            "Investment Amount",
            min_value=1_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
            help="Portfolio simulations and backtests will be expressed in this monetary amount.",
        )
        mgmt_fee_annual = management_fee_from_wealth(investment_amount)
        st.caption(
            f"Estimated annual management fee: **{mgmt_fee_annual:.2%}** "
            "(applied pro rata on a monthly basis)."
        )

    with colC:
        investment_horizon_years = st.selectbox(
            "Investment Horizon",
            options=[1, 2, 3, 5, 7, 10],
            index=0,
            format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
        )

    with colD:
        rebalance_label = st.selectbox(
            "Rebalancing Frequency",
            options=["Yearly", "Quarterly", "Monthly"],
            index=0,
        )
        if rebalance_label == "Yearly":
            rebalancing = 12
        elif rebalance_label == "Quarterly":
            rebalancing = 3
        else:
            rebalancing = 1

    # ---- Full-width estimation window block (no column) ----
    st.subheader("Estimation Window")

    use_custom_est = st.checkbox(
        "Enable custom estimation window",
        value=False,
        help=(
            "By default, the model uses 12 months of historical data to estimate "
            "expected returns and risk. Enable only if you understand the impact "
            "on estimation error and model stability."
        ),
    )

    if not use_custom_est:
        est_months = 12
        st.info("Using default setting: **12-month (1-year) estimation window**.")
    else:
        est_months = st.selectbox(
            "Select estimation window (in months)",
            options=[6, 12, 24, 36, 60],
            index=1,  # 12 months as the suggested default
            format_func=lambda m: f"{m} months",
        )

        st.warning(
            "**Caution:** Changing the estimation window alters the balance between "
            "**statistical reliability** and **reactivity to market regimes**. "
            "Shorter windows make the model more sensitive to recent moves but also "
            "more exposed to noise and unstable covariance estimates. Longer windows "
            "smooth short-term noise but may overweight outdated market conditions."
        )

    st.markdown("---")

    # ============================================================
    # STEP 2 – UNIVERSE & FILTERS
    # ============================================================
    st.markdown("### Step 2 – Universe & Filters")

    # --- Get metadata for chosen equity universe and other assets ---
    if universe_choice == "SP500":
        metadata_equity = data["metadata"]["SP500"]
    else:
        metadata_equity = data["metadata"]["MSCI"]

    metadata_other = data["metadata"]["Other"]

    # ---------- 2.1 Equity filters: sectors & ESG ----------
    st.subheader("Equity Filters")

    sectors_available = (
        metadata_equity["SECTOR"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    col_sect, col_esg = st.columns(2)

    with col_sect:
        selected_sectors = st.multiselect(
            "Sectors to include in equity universe",
            options=sectors_available,
            default=[],
            help="If you select all sectors, no sector filter is applied.",
        )

        if len(selected_sectors) == len(sectors_available) or len(selected_sectors) == 0:
            keep_sectors = None
        else:
            keep_sectors = selected_sectors

    with col_esg:
        esg_options = ["L", "M", "H"]
        selected_esg = st.multiselect(
            "ESG categories to include",
            options=esg_options,
            default=[],
            help="L = Low, M = Medium, H = High. Selecting all applies no ESG filter.",
        )

        if len(selected_esg) == len(esg_options) or len(selected_esg) == 0:
            keep_esg = None
        else:
            keep_esg = selected_esg

    # ---------- 2.2 Other asset classes & instruments ----------
    st.subheader("Other Asset Classes")

    asset_classes_all = (
        metadata_other["ASSET_CLASS"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_asset_classes_other = st.multiselect(
        "Asset classes to include in the universe (beyond equity)",
        options=asset_classes_all,
        default=[],
        help=(
            "These asset classes will be available to the optimizer. "
            "Constraints later control how much can be allocated to each."
        ),
    )

    keep_ids_by_class = {}

    for ac in selected_asset_classes_other:
        subset = metadata_other[metadata_other["ASSET_CLASS"] == ac]
        ids_in_class = subset.index.astype(str).tolist()

        # Build pretty labels: TICKER – NAME  (fallbacks if missing)
        label_map: dict[str, str] = {}
        for idx, row in subset.iterrows():
            key = str(idx)

            ticker = None
            name = None
            if "TICKER" in subset.columns and pd.notna(row.get("TICKER")):
                ticker = str(row["TICKER"]).strip()
            if "NAME" in subset.columns and pd.notna(row.get("NAME")):
                name = str(row["NAME"]).strip()

            if ticker and name:
                label_map[key] = f"{ticker} – {name}"
            elif name:
                label_map[key] = name
            elif ticker:
                label_map[key] = ticker
            else:
                label_map[key] = key  # fallback: just the ID

        st.markdown(f"**{ac} instruments to include**")
        selected_ids = st.multiselect(
            f"Select {ac} instruments (leave all selected to keep full class)",
            options=ids_in_class,
            default=[],
            format_func=lambda x: label_map.get(str(x), str(x)),
        )

        # Only store a filter if the user actually deselected something
        if 0 < len(selected_ids) < len(ids_in_class):
            keep_ids_by_class[ac] = selected_ids

    keep_ids_by_class = keep_ids_by_class if keep_ids_by_class else None

    st.markdown("---")

    # ============================================================
    # STEP 3 – RISK PROFILE QUESTIONNAIRE → GAMMA
    # ============================================================
    st.markdown("### Step 3 – Risk Profile Questionnaire")

    st.caption(
        "Answer each question on a 1–5 scale. "
        "1 = very conservative, 5 = very aggressive."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        q1 = st.slider(
            "1. Reaction to a -20% loss in one year\n"
            "1 = sell everything, 5 = buy more",
            min_value=1, max_value=5, value=3,
        )

        q2 = st.slider(
            "2. Comfort with large fluctuations\n"
            "1 = not at all, 5 = very comfortable",
            min_value=1, max_value=5, value=3,
        )

        q3 = st.slider(
            "3. Return vs risk trade-off\n"
            "1 = stable low returns, 5 = max return even with large risk",
            min_value=1, max_value=5, value=3,
        )

        q4 = st.slider(
            "4. Investment horizon\n"
            "1 = < 1 year, 5 = > 10 years",
            min_value=1, max_value=5, value=3,
        )

        q5 = st.slider(
            "5. How do you view risk?\n"
            "1 = something to avoid, 5 = essential for higher returns",
            min_value=1, max_value=5, value=3,
        )

    with col_right:
        q6 = st.slider(
            "6. Stress during market crashes\n"
            "1 = extremely stressed, 5 = not stressed at all",
            min_value=1, max_value=5, value=3,
        )

        q7 = st.slider(
            "7. Stability of your income/finances\n"
            "1 = very unstable, 5 = very stable",
            min_value=1, max_value=5, value=3,
        )

        q8 = st.slider(
            "8. Experience with investing\n"
            "1 = not familiar, 5 = very experienced",
            min_value=1, max_value=5, value=3,
        )

        q9 = st.slider(
            "9. Reaction to a +20% gain in one year\n"
            "1 = sell to lock gains, 5 = add significantly more money",
            min_value=1, max_value=5, value=3,
        )

        q10 = st.slider(
            "10. Share of net worth in risky assets\n"
            "1 = < 10%, 5 = > 60%",
            min_value=1, max_value=5, value=3,
        )

    scores = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    S = sum(scores)
    gamma = 0.5 + 0.15 * (S - 10)  # internal only

    if S <= 20:
        profile_label = "Very Conservative"
        profile_text = (
            "You have a **very low tolerance for risk** and prefer capital preservation. "
            "The portfolio will be tilted towards safer, lower-volatility assets."
        )
    elif S <= 30:
        profile_label = "Conservative"
        profile_text = (
            "You are **cautious with risk**, but willing to accept some fluctuations. "
            "The portfolio will prioritize stability with a moderate growth component."
        )
    elif S <= 35:
        profile_label = "Balanced"
        profile_text = (
            "You have a **balanced attitude** towards risk and return. "
            "The portfolio will mix growth assets with stabilizing components."
        )
    elif S <= 42:
        profile_label = "Dynamic"
        profile_text = (
            "You are **comfortable with risk** and seek higher returns. "
            "The portfolio will have a strong allocation to growth and risky assets."
        )
    else:
        profile_label = "Aggressive"
        profile_text = (
            "You have a **high risk tolerance** and focus on return maximization. "
            "The portfolio will be heavily exposed to volatile, return-seeking assets."
        )

    st.markdown("")
    col_score, col_profile = st.columns(2)
    with col_score:
        st.metric("Total Risk Score (S)", f"{S} / 50")
    with col_profile:
        st.markdown(f"**Risk Profile:** {profile_label}")
        st.caption(profile_text)

    st.markdown("---")

    # ============================================================
    # STEP 4 – CONSTRAINTS
    # ============================================================
    st.markdown("### Step 4 – Constraints")

    st.caption(
        "All constraints are expressed as **fractions** (0.10 = 10%). "
        "Leave min = 0 and max = 1 to avoid imposing a constraint."
    )

    # ------------------------------------------------------------
    # 4.1 Max weight per asset (with safe default + warning)
    # ------------------------------------------------------------
    st.subheader("Maximum Weight per Asset")

    use_custom_max = st.checkbox(
        "Enable custom maximum weight per asset",
        value=False,
        help="By default, each asset is capped at 5%. Enable only if you understand concentration risk."
    )

    if not use_custom_max:
        max_weight_per_asset = 0.05
        st.info("Using default limit: **5% maximum per individual asset**.")
    else:
        max_weight_per_asset = st.slider(
            "Select maximum weight per asset",
            min_value=0.01,
            max_value=0.25,
            value=0.05,
            step=0.01,
            help="Higher caps increase concentration risk and may reduce diversification."
        )

        st.warning(
            "**Caution:** Increasing the maximum weight per asset may significantly raise your "
            "**idiosyncratic risk** and reduce the portfolio's **diversification benefits**. "
            "Large individual exposures can amplify the impact of adverse movements in a single "
            "security, especially during periods of market stress."
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.2 Sector constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity Sector Constraints (relative to the equity exposure)")

    if keep_sectors is None:
        sectors_for_constraints = sectors_available
    else:
        sectors_for_constraints = keep_sectors

    sector_constraints = {}
    sector_min_budget = 0.0  # sum of mins so far, must stay <= 1

    for sec in sectors_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - sector_min_budget)

        with st.expander(f"{sec}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                sec_min = st.number_input(
                    f"Min share of {sec} in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_min_{sec}",
                )

            # update budget after this min
            sector_min_budget += sec_min

            with col_max:
                sec_max = st.number_input(
                    f"Max share of {sec} in Equity",
                    min_value=0.0,  # ensures min <= max
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_max_{sec}",
                )

            # Professional-style warnings at boundaries
            eps = 1e-8
            if remaining_min_budget > 0 and abs(sec_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **{sec}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of sector minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if sec_min > 0 and abs(sec_max - sec_min) < eps:
                st.info(
                    f"For **{sec}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer to rebalance within this sector."
                )

        cons = {}
        if sec_min > 0:
            cons["min"] = float(sec_min)
        if sec_max < 1.0:
            cons["max"] = float(sec_max)
        if cons:
            sector_constraints[sec] = cons

    if not sector_constraints:
        sector_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.3 ESG constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity ESG Score Constraints (relative to the equity exposure)")

    esg_all_labels = ["L", "M", "H"]
    if keep_esg is None:
        esg_for_constraints = esg_all_labels
    else:
        esg_for_constraints = keep_esg

    esg_constraints = {}
    esg_min_budget = 0.0

    for label in esg_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - esg_min_budget)

        with st.expander(f"ESG {label}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                esg_min = st.number_input(
                    f"Min share of ESG {label} score in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_min_{label}",
                )

            esg_min_budget += esg_min

            with col_max:
                esg_max = st.number_input(
                    f"Max share ESG {label} score in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_max_{label}",
                )

            eps = 1e-8
            if remaining_min_budget > 0 and abs(esg_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **ESG {label}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of ESG minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if esg_min > 0 and abs(esg_max - esg_min) < eps:
                st.info(
                    f"For **ESG {label}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer within this ESG bucket."
                )

        cons = {}
        if esg_min > 0:
            cons["min"] = float(esg_min)
        if esg_max < 1.0:
            cons["max"] = float(esg_max)
        if cons:
            esg_constraints[label] = cons

    if not esg_constraints:
        esg_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.4 Asset-class constraints (total portfolio)
    # ------------------------------------------------------------
    st.subheader("Asset-Class Constraints (total portfolio)")

    if not selected_asset_classes_other:
        st.info(
            "You have selected an **equity-only universe**. "
            "By construction, 100% of the portfolio will be invested in Equity."
        )
        asset_class_constraints = None

    else:

        asset_classes_for_constraints = ["Equity"] + selected_asset_classes_other

        asset_class_constraints = {}
        ac_min_budget = 0.0

        for ac in asset_classes_for_constraints:
            remaining_min_budget = max(0.0, 1.0 - ac_min_budget)

            with st.expander(f"{ac}", expanded=False):
                col_min, col_max = st.columns(2)

                with col_min:
                    ac_min = st.number_input(
                        f"Min portfolio weight in {ac}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f",
                        key=f"ac_min_{ac}",
                    )

                ac_min_budget += ac_min

                with col_max:
                    ac_max = st.number_input(
                        f"Max portfolio weight in {ac}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.05,
                        format="%.2f",
                        key=f"ac_max_{ac}",
                    )

                eps = 1e-8
                if remaining_min_budget > 0 and abs(ac_min - remaining_min_budget) < eps:
                    st.warning(
                        f"The minimum allocation entered for **{ac}** is at the upper feasible bound. "
                        "Any higher minimum would force the sum of asset-class minima above **100% of the portfolio** "
                        "and is therefore not admissible."
                    )

                if ac_min > 0 and abs(ac_max - ac_min) < eps:
                    st.info(
                        f"For **{ac}**, the minimum and maximum allocations are effectively identical. "
                        "This leaves no flexibility for the optimizer to reallocate across asset classes."
                    )

            cons = {}
            if ac_min > 0:
                cons["min"] = float(ac_min)
            if ac_max < 1.0:
                cons["max"] = float(ac_max)
            if cons:
                asset_class_constraints[ac] = cons

        if not asset_class_constraints:
            asset_class_constraints = None

        st.markdown("---")

    constraint_errors = validate_constraints(
        sector_constraints=sector_constraints,
        esg_constraints=esg_constraints,
        asset_class_constraints=asset_class_constraints,
    )

    if constraint_errors:
        st.error("The current constraint configuration is not feasible:")
        for msg in constraint_errors:
            st.write(f"• {msg}")


    # ============================================================
    # STEP 5 – RUN OPTIMIZATION & BACKTEST
    # ============================================================
    st.markdown("### Step 5 – Run Optimization & Backtest")

    run_clicked = st.button(
        "Run Optimization & Backtest",
        type="primary",
        disabled=bool(constraint_errors),
    )

    if run_clicked:
        # 1) Check constraints first
        if constraint_errors:
            st.error("The current constraint configuration is not feasible:")
            for msg in constraint_errors:
                st.write(f"• {msg}")
            st.stop()  # do not run the optimizer

        # 2) Build config only if constraints are okay
        config = PortfolioConfig(
            today_date=pd.Timestamp("2025-10-01"),
            investment_horizon_years=investment_horizon_years,
            est_months=est_months,
            rebalancing=rebalancing,
            gamma=gamma,
            universe_choice=universe_choice,
            keep_sectors=keep_sectors,
            keep_esg=keep_esg,
            selected_asset_classes_other=selected_asset_classes_other,
            keep_ids_by_class=keep_ids_by_class,
            max_weight_per_asset=max_weight_per_asset,
            sector_constraints=sector_constraints,
            esg_constraints=esg_constraints,
            asset_class_constraints=asset_class_constraints,
            initial_wealth=investment_amount
        )

        # 3) Run engine with friendly error handling
        try:
            with st.spinner("Optimizing and backtesting..."):
                perf, summary_df, debug_weights_df = run_backtest(config, data)
                today_res = run_today_optimization(config, data)
        except ValueError as e:
            # This catches "Optimization failed: Positive directional derivative..." etc.
            st.error(
                "The optimizer could not find a feasible portfolio with the current set of "
                "constraints and per-asset limits."
            )
            st.caption(
                "This typically happens when minimum allocations across sectors, ESG buckets or "
                "asset classes are too tight relative to the available universe and the maximum "
                "weight per asset. Please relax some minimum constraints or increase the maximum "
                "weight per asset, then try again."
            )
            # Optional: show the raw technical message for yourself
            # st.text(f"Technical details: {e}")
            st.stop()

        st.success("Optimization completed.")

        st.session_state["backtest_results"] = {
            "perf": perf,
            "summary_df": summary_df,
            "debug_weights_df": debug_weights_df,
            "today_res": today_res,
            "investment_amount": investment_amount,
            "universe_choice": universe_choice,
            "investment_horizon_years": investment_horizon_years,
            "est_months": est_months,
            "rebalancing": rebalancing,
            "gamma": gamma,
            "profile_label": profile_label,
            "max_weight_per_asset": max_weight_per_asset,
            "selected_asset_classes_other": selected_asset_classes_other,
            "sector_constraints": sector_constraints,
            "esg_constraints": esg_constraints,
            "asset_class_constraints": asset_class_constraints,
        }

    if "backtest_results" in st.session_state:
        r = st.session_state["backtest_results"]

        perf = r["perf"]
        summary_df = r["summary_df"]
        debug_weights_df = r["debug_weights_df"]
        today_res = r["today_res"]

        investment_amount = r["investment_amount"]
        universe_choice = r["universe_choice"]
        investment_horizon_years = r["investment_horizon_years"]
        est_months = r["est_months"]
        rebalancing = r["rebalancing"]
        gamma = r["gamma"]
        profile_label = r["profile_label"]
        max_weight_per_asset = r["max_weight_per_asset"]
        selected_asset_classes_other = r["selected_asset_classes_other"]
        sector_constraints = r["sector_constraints"]
        esg_constraints = r["esg_constraints"]
        asset_class_constraints = r["asset_class_constraints"]


        tab_backtest, tab_today = st.tabs(["📈 Backtest", "📌 Today's Portfolio"])

        with tab_backtest:
            st.subheader("Backtest Performance")

            if not perf.empty:
                # 1) Compute backtest stats (for max drawdown, etc.)
                stats = compute_backtest_stats(perf)

                # --------------------------------------------------------
                # A) BUILD DATAFRAME WITH PORTFOLIO + BENCHMARKS (CUMRET)
                # --------------------------------------------------------
                returns_bench = data.get("benchmarks", None)

                if returns_bench is not None and not returns_bench.empty:
                    bench = returns_bench.reindex(perf.index)
                    bench_cum = (1.0 + bench).cumprod() - 1.0
                    combined = pd.concat([perf["CumReturn"], bench_cum], axis=1)
                    combined.columns = ["Portfolio"] + list(bench_cum.columns)
                else:
                    combined = pd.DataFrame({"Portfolio": perf["CumReturn"]})

                # Convert index to timestamp
                if isinstance(combined.index, pd.PeriodIndex):
                    combined.index = combined.index.to_timestamp()

                chart_data = combined.reset_index().rename(columns={"Date": "Date"})

                import altair as alt
                chart_data_long = chart_data.melt("Date", var_name="Series", value_name="Return")

                # --------------------------------------------------------
                # B) CUMULATIVE RETURN CHART WITH BENCHMARKS
                # --------------------------------------------------------
                st.markdown("**Cumulative Return of the Strategy vs Benchmarks**")

                max_ret = float(chart_data_long["Return"].max()) if not chart_data_long["Return"].isna().all() else 0.0
                max_ret = max(max_ret, 0.0)
                max_tick = (int(max_ret * 10) + 1) / 10.0 if max_ret > 0 else 0.1
                tick_values = [i / 10.0 for i in range(0, int(max_tick * 10) + 1)]

                base_ret = (
                    alt.Chart(chart_data_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y(
                            "Return:Q",
                            title="Cumulative return",
                            scale=alt.Scale(domain=[0, max_tick], nice=False),
                            axis=alt.Axis(format="%", values=tick_values),
                        ),
                        color=alt.Color("Series:N", sort=["Portfolio", "S&P 500", "MSCI WORLD"]),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date", format="%b %Y"),
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Return:Q", title="Cumulative return", format=".2%"),
                        ],
                    )
                )

                # Drawdown vertical lines
                if stats and stats["max_drawdown_start"] is not None:
                    dd_start, dd_end = stats["max_drawdown_start"], stats["max_drawdown_end"]
                    vline_data = pd.DataFrame({"Date": [dd_start, dd_end], "Label": ["DD start", "DD end"]})

                    vlines = (
                        alt.Chart(vline_data)
                        .mark_rule(color="red", strokeDash=[4, 4], size=2)
                        .encode(x="Date:T")
                    )

                    chart_ret = alt.layer(base_ret, vlines).interactive()
                else:
                    chart_ret = base_ret.interactive()

                st.altair_chart(chart_ret, use_container_width=True)

                st.markdown(
                    "<span style='color:red;'>Red dashed vertical lines indicate the "
                    "<b>start</b> and <b>end</b> of the worst drawdown observed over the backtest period.</span>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # --------------------------------------------------------
                # C) SECOND CHART – PORTFOLIO WEALTH
                # --------------------------------------------------------
                st.markdown("**Evolution of Portfolio Wealth**")

                perf_plot = perf.copy()
                if isinstance(perf_plot.index, pd.PeriodIndex):
                    perf_plot.index = perf_plot.index.to_timestamp()
                perf_plot = perf_plot.reset_index()

                max_wealth = float(perf["Wealth"].max())
                upper_wealth = max(max_wealth, 1.0) * 1.05

                base_wealth = (
                    alt.Chart(perf_plot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y("Wealth:Q", scale=alt.Scale(domain=[0, upper_wealth])),
                        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Wealth:Q", format=",.0f")],
                    )
                )

                if stats and stats["max_drawdown_start"] is not None:
                    chart_wealth = alt.layer(base_wealth, vlines).interactive()
                else:
                    chart_wealth = base_wealth.interactive()

                st.altair_chart(chart_wealth, use_container_width=True)

                # --------------------------------------------------------
                # D) BACKTEST STATISTICS TABLE
                # --------------------------------------------------------
                st.markdown("**Backtest Statistics**")

                initial = investment_amount
                final_wealth = investment_amount * float(perf["Growth"].iloc[-1])

                def fmt_pct(x):
                    return f"{x:.2%}"

                stats_rows = [
                    ("Initial invested wealth", f"{initial:,.0f}"),
                    ("Final wealth at end of backtest", f"{final_wealth:,.0f}"),
                    ("Annualised average return", fmt_pct(stats["annualised_avg_return"])),
                    ("Annualised volatility", fmt_pct(stats["annualised_volatility"])),
                    ("Annualised cumulative return", fmt_pct(stats["annualised_cum_return"])),
                    ("Min monthly return", fmt_pct(stats["min_monthly_return"])),
                    ("Max monthly return", fmt_pct(stats["max_monthly_return"])),
                    ("Max drawdown", fmt_pct(stats["max_drawdown"])),
                    ("Max drawdown start", stats["max_drawdown_start"].strftime("%b %Y")),
                    ("Max drawdown end", stats["max_drawdown_end"].strftime("%b %Y")),
                    ("Max drawdown duration", f"{stats['max_drawdown_duration_months']} months"),
                ]

                st.table(pd.DataFrame(stats_rows, columns=["Metric", "Value"]))

                # --------------------------------------------------------
                # E) AI Commentary on the Backtest
                # --------------------------------------------------------
                st.markdown("### AI Commentary on Backtest Results")

                explain_btn = st.button(
                    "Generate AI Commentary on Backtest",
                    type="secondary",
                    help="Ask the Phi Investment Capital digital assistant to provide a "
                         "client-friendly interpretation of the backtest results.",
                )

                if explain_btn:
                    client_llm = get_llm_client()

                    # Build a textual context for the model, including client inputs
                    context_text = build_backtest_context_text(
                        stats=stats,
                        perf=perf,
                        investment_amount=investment_amount,
                        universe_choice=universe_choice,
                        investment_horizon_years=investment_horizon_years,
                        est_months=est_months,
                        rebalancing=rebalancing,
                        gamma=gamma,
                        profile_label=profile_label,
                        max_weight_per_asset=max_weight_per_asset,
                        selected_asset_classes_other=selected_asset_classes_other,
                        sector_constraints=sector_constraints,
                        esg_constraints=esg_constraints,
                        asset_class_constraints=asset_class_constraints,
                    )

                    system_prompt = (
                        "You are a digital investment assistant for Phi Investment Capital. "
                        "You are given a summary of a client's configuration and portfolio backtest. "
                        "Provide a professional, client-friendly commentary on the results.\n\n"
                        "Guidelines:\n"
                        "- Refer to the client's risk profile, constraints, and investment horizon when relevant.\n"
                        "- Comment on constraints the client choose"
                        "- Comment on the balance between return and risk (volatility and drawdowns).\n"
                        "- Highlight any notable features of the drawdown profile and overall behaviour over time.\n"
                        "- You may mention that tighter constraints or lower max weights can limit performance but improve diversification.\n"
                        "- Do NOT give investment recommendations or instructions to buy/sell.\n"
                        "- Do NOT make promises about future performance.\n"
                        "- Keep the answer to about 2–5 short paragraphs, in a calm and professional tone."
                    )

                    user_message = (
                        "Here is the full context (client configuration and backtest summary). "
                        "Please provide a concise commentary for the client:\n\n"
                        f"{context_text}"
                    )

                    with st.spinner("Generating AI commentary..."):
                        response = client_llm.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                        )
                        commentary = response.choices[0].message.content

                    st.markdown(
                        """
                        **Phi Investment Capital – Backtest Commentary**  
                        *(Generated by the digital assistant based on your inputs and the statistics above.)*
                        """
                    )
                    st.markdown(commentary)


            else:
                st.warning("No valid backtest window for the selected settings.")

        with tab_today:
            st.subheader("Today's Optimal Portfolio")

            today_df = today_res["weights"]
            top5 = today_res["top5"]
            alloc_by_ac = today_res["alloc_by_asset_class"]
            sector_in_eq = today_res["sector_in_equity"]
            esg_in_eq = today_res["esg_in_equity"]

            st.markdown("**Top 5 Holdings**")
            st.dataframe(top5)


            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown("**By Asset Class**")

                if not alloc_by_ac.empty:
                    # 1️⃣ Build base DataFrame
                    df_ac = alloc_by_ac.reset_index()
                    df_ac.columns = ["AssetClass", "Weight"]

                    # 2️⃣ Add asset-class min/max constraints (if any)
                    df_ac["Min"] = df_ac["AssetClass"].map(
                        lambda ac: asset_class_constraints.get(ac, {}).get("min", None)
                        if asset_class_constraints
                        else None
                    )
                    df_ac["Max"] = df_ac["AssetClass"].map(
                        lambda ac: asset_class_constraints.get(ac, {}).get("max", None)
                        if asset_class_constraints
                        else None
                    )

                    # 3️⃣ Base blue bars
                    bars_ac = (
                        alt.Chart(df_ac)
                        .mark_bar(color="#4BA3FF")
                        .encode(
                            x=alt.X("AssetClass:N", title=""),
                            y=alt.Y("Weight:Q", title="Portfolio weight"),
                            tooltip=[
                                alt.Tooltip("AssetClass:N", title="Asset class"),
                                alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                            ],
                        )
                    )

                    # 4️⃣ YEllOW tick for MIN constraint
                    min_marks_ac = (
                        alt.Chart(df_ac)
                        .mark_tick(
                            orient="horizontal",
                            color="yellow",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="AssetClass:N",
                            y="Min:Q",
                            tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                        )
                        .transform_filter("datum.Min != null")
                    )

                    # 5️⃣ RED tick for MAX constraint
                    max_marks_ac = (
                        alt.Chart(df_ac)
                        .mark_tick(
                            orient="horizontal",
                            color="red",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="AssetClass:N",
                            y="Max:Q",
                            tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                        )
                        .transform_filter("datum.Max != null")
                    )

                    # 6️⃣ Combine layers
                    chart_ac = (bars_ac + min_marks_ac + max_marks_ac).properties(
                        height=300
                    ).interactive()

                    st.altair_chart(chart_ac, use_container_width=True)

                else:
                    st.info("No allocation across asset classes.")

            with colB:
                st.markdown("**Sector Breakdown (Equity)**")

                if not sector_in_eq.empty:

                    # 1️⃣ Build base DataFrame for chart
                    df_sector = sector_in_eq.reset_index()
                    df_sector.columns = ["Sector", "Weight"]

                    # 2️⃣ Add sector min/max constraints (if any)
                    df_sector["Min"] = df_sector["Sector"].map(
                        lambda s: sector_constraints.get(s, {}).get("min", None) if sector_constraints else None
                    )
                    df_sector["Max"] = df_sector["Sector"].map(
                        lambda s: sector_constraints.get(s, {}).get("max", None) if sector_constraints else None
                    )

                    # 3️⃣ Base bar chart
                    bars = (
                        alt.Chart(df_sector)
                        .mark_bar(color="#4BA3FF")
                        .encode(
                            x=alt.X("Sector:N", title="", sort="-y"),
                            y=alt.Y("Weight:Q", title="Weight in Equity"),
                            tooltip=[
                                alt.Tooltip("Sector:N"),
                                alt.Tooltip("Weight:Q", format=".2%"),
                            ],
                        )
                    )

                    # 4️⃣ YELLOW tick for MIN constraint
                    min_marks = (
                        alt.Chart(df_sector)
                        .mark_tick(
                            orient="horizontal",
                            color="yellow",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="Sector:N",
                            y="Min:Q",
                            tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                        )
                        .transform_filter("datum.Min != null")
                    )

                    # 5️⃣ RED tick for MAX constraint
                    max_marks = (
                        alt.Chart(df_sector)
                        .mark_tick(
                            orient="horizontal",
                            color="red",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="Sector:N",
                            y="Max:Q",
                            tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                        )
                        .transform_filter("datum.Max != null")
                    )

                    # 6️⃣ Combine layers
                    chart_sector = (bars + min_marks + max_marks).properties(
                        height=300
                    ).interactive()

                    st.altair_chart(chart_sector, use_container_width=True)

                else:
                    st.info("No equity allocation.")

            with colC:
                st.markdown("**ESG Breakdown (Equity)**")

                if not esg_in_eq.empty:

                    # 1️⃣ Build base DataFrame
                    df_esg = esg_in_eq.reset_index()
                    df_esg.columns = ["ESG", "Weight"]

                    # 2️⃣ Add ESG min/max constraints (if any)
                    df_esg["Min"] = df_esg["ESG"].map(
                        lambda s: esg_constraints.get(s, {}).get("min", None) if esg_constraints else None
                    )
                    df_esg["Max"] = df_esg["ESG"].map(
                        lambda s: esg_constraints.get(s, {}).get("max", None) if esg_constraints else None
                    )

                    # 3️⃣ Base blue bars
                    bars_esg = (
                        alt.Chart(df_esg)
                        .mark_bar(color="#4BA3FF")
                        .encode(
                            x=alt.X("ESG:N", title=""),
                            y=alt.Y("Weight:Q", title="Weight in Equity"),
                            tooltip=[
                                alt.Tooltip("ESG:N"),
                                alt.Tooltip("Weight:Q", format=".2%"),
                            ],
                        )
                    )

                    # 4️⃣ YELLOW tick for MIN constraint
                    min_marks_esg = (
                        alt.Chart(df_esg)
                        .mark_tick(
                            orient="horizontal",
                            color="yellow",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="ESG:N",
                            y="Min:Q",
                            tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                        )
                        .transform_filter("datum.Min != null")
                    )

                    # 5️⃣ RED tick for MAX constraint
                    max_marks_esg = (
                        alt.Chart(df_esg)
                        .mark_tick(
                            orient="horizontal",
                            color="red",
                            size=40,
                            thickness=3,
                        )
                        .encode(
                            x="ESG:N",
                            y="Max:Q",
                            tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                        )
                        .transform_filter("datum.Max != null")
                    )

                    # 6️⃣ Combine everything
                    chart_esg = (bars_esg + min_marks_esg + max_marks_esg).properties(
                        height=300
                    ).interactive()

                    st.altair_chart(chart_esg, use_container_width=True)

                else:
                    st.info("No equity allocation.")

            st.markdown(
                "<span style='color:#DAA520;'>Yellow horizontal markers</span> "
                "indicate **minimum allocation bounds** (lower weights), while "
                "<span style='color:red;'>red horizontal markers</span> indicate "
                "**maximum allocation limits** for each sector, ESG bucket or asset class.",
                unsafe_allow_html=True,
            )

            with st.expander("Full Portfolio Weights"):
                st.dataframe(today_df)


# --------------- PAGE 3: PORTFOLIO OPTIMIZATION ---------------
def page_ai_assistant():
    st.title("Phi Assistant ")

    st.markdown(
        """
        Our AI-powered assistant is designed to enhance your experience on our portfolio management platform. 
        It can support you across three key areas.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Platform Guidance & Functionality")
        st.markdown(
            """
            Navigate the application with confidence:

            The assistant can explain each section of the site — 
            from selecting your investment universe to configuring 
            constraints, interpreting backtests, and reviewing today’s 
            optimal portfolio. Whether you're unsure about a step or want to
            understand how a specific feature works, it provides clear, client-friendly guidance.
            """
        )

    with col2:
        st.subheader("2. Financial & Theoretical Concepts")
        st.markdown(
            """
            Understand the rationale behind your portfolio.:

            Ask about diversification, risk/return trade-offs, gamma (risk aversion), the Markowitz optimization 
            framework, or the meaning of any chart or metric shown in the application. The assistant provides clear 
            explanations grounded in quantitative finance — always educational, never advisory.
            """
        )

    with col3:
        st.subheader("3. Market Context & Current Themes")
        st.markdown(
            """
            Stay informed about what’s happening in the financial world.:

            You can request neutral, factual insights about current market conditions, macroeconomic themes, 
            or asset-class developments. The assistant offers high-level context to help you make sense of the 
            broader environment in which portfolios operate.  
            """
        )

    st.markdown(
        """
        ⚠️ **Important:** The assistant provides general information and educational insights only.
        It does **not** offer personalized investment recommendations or specific trading advice.
        """
    )

    client = get_llm_client()


    # Initialize chat history in session_state
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "system",
                "content": (
                    "You are a digital investment assistant for an asset and risk management firm named "
                    "Phi Investment Capital. You are integrated into a web application. Your name is Phi Assiatant\n\n"
                    "Your primary role is to support clients in understanding:\n"
                    "- How the app works (its pages, steps, and main functionalities)\n"
                    "- The meaning of inputs (risk questionnaire, constraints, filters, universe choices)\n"
                    "- The outputs of the optimizer (performance charts, drawdowns, asset-class/sector/ESG breakdowns)\n"
                    "- General investment concepts related to diversification, risk/return, and portfolio construction\n"
                    "- High-level, factual information about financial markets and asset classes\n\n"
                    "Context about the app's functionality:\n"
                    "- The app has an 'About us' page describing Phi Investment Capital and its quantitative approach.\n"
                    "- The 'Portfolio optimization' page guides the client through:\n"
                    "  1) General settings: equity universe (S&P 500 or MSCI World), investment amount, horizon,\n"
                    "     rebalancing frequency, and estimation window.\n"
                    "  2) Universe & filters: sector and ESG filters for equities, plus selection of other asset classes\n"
                    "     (e.g. fixed income, commodities, alternatives) and instruments within them.\n"
                    "  3) A risk profile questionnaire (10 questions) that produces a risk score and an internal\n"
                    "     risk aversion parameter called gamma.\n"
                    "  4) Constraints: maximum weight per asset, sector constraints, ESG constraints, and asset-class\n"
                    "     constraints at the total portfolio level.\n"
                    "  5) Optimization & backtest: a Markowitz mean–variance long-only optimization with constraints,\n"
                    "     followed by a backtest showing cumulative returns, portfolio wealth, drawdowns, and summary statistics.\n"
                    "- The app also shows today's optimal portfolio with:\n"
                    "  - Top holdings\n"
                    "  - Allocation by asset class\n"
                    "  - Sector breakdown within equity\n"
                    "  - ESG breakdown within equity\n\n"
                    "How you should behave:\n"
                    "- When clients ask about functionality, clearly explain which step of the process it relates to and\n"
                    "  describe what the app does in that step (without assuming you see their exact numbers).\n"
                    "- When clients ask about charts or metrics, explain conceptually what they represent (e.g. cumulative return,\n"
                    "  max drawdown, annualised volatility, wealth evolution).\n"
                    "- When clients ask about constraints or filters, explain how they influence diversification, risk concentration,\n"
                    "  and the optimizer's feasible set.\n"
                    "- When clients ask about financial markets, provide neutral, factual, and high-level explanations only.\n\n"
                    "Compliance and limitations:\n"
                    "- Do NOT provide personalized investment advice or recommendations.\n"
                    "- Do NOT tell clients what they should buy, sell, or hold.\n"
                    "- Do NOT give specific portfolio allocations, target returns, or forecasts for individual securities.\n"
                    "- You may explain trade-offs (e.g. higher risk vs higher potential return) in general terms.\n"
                    "- Keep a professional, calm, and client-oriented tone, as a relationship manager in a wealth management\n"
                    "  or asset management firm would, but always focus on education and explanation rather than advice."
                ),
            }
        ]

    # Show previous messages (except system)
    for msg in st.session_state.ai_messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question")
    if user_input:
        # 1) Add user message to history
        st.session_state.ai_messages.append({"role": "user", "content": user_input})

        # 2) Display user bubble
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3) Call OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=st.session_state.ai_messages,
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        # 4) Save assistant reply in history
        st.session_state.ai_messages.append(
            {"role": "assistant", "content": reply}
        )






if __name__ == "__main__":
    main()
