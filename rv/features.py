"""Step 7: the 97 input variables of Table A.1.

Naming
  *_wr : 52 ratios from the WRDS Financial Ratios Suite
  *_an : 39 sorting variables from the anomaly literature
  other: 4 size variables + CAPMbeta + industry
l_x is the same quarter of the prior year and d_x = x - l_x (see compustat.py).
Flow items (sale, ib, cogs ...) are sums over the last four quarters (TTM).
"""
import numpy as np
import pandas as pd


def _div(a, b, positive=False):
    """Safe division: NaN when the denominator is 0 (or <= 0 when positive=True)."""
    b = pd.Series(b, copy=False)
    bad = (b <= 0) if positive else (b == 0)
    return a / b.mask(bad)


def _z(s):
    """Treat missing as 0 (the cond(missing(x),0,x) / rowtotal convention in Table A.1)."""
    return s.fillna(0)


# ---------- 52 WRDS ratios ----------
WR = {
    "Accrual_wr":        lambda d: _div(d["ocf"] - d["ib"], d["at"]),
    "aftret_eq_wr":      lambda d: _div(d["ibcom"], d["ceq"]),
    "aftret_equity_wr":  lambda d: _div(d["ib"], d["seq"]),
    "aftret_invcapx_wr": lambda d: _div(d["ib"] + d["xint"] + _z(d["mii"]),
                                        d["l_icapt"] + _z(d["l_txditc"]) - _z(d["l_mib"]), positive=True),
    "at_turn_wr":        lambda d: _div(d["sale"], d["at"]),
    "capital_ratio_wr":  lambda d: _div(d["dltt"], d["dltt"] + d["ceq"] + _z(d["pstk"])),
    "cash_conversion_wr":lambda d: (_div((d["invt"] + d["l_invt"]) / 2, d["cogs"] / 365)
                                    + _div((d["rect"] + d["l_rect"]) / 2, d["sale"] / 365)
                                    - _div((d["ap"] + d["l_ap"]) / 2, d["cogs"] / 365)),
    "cash_debt_wr":      lambda d: _div(d["ocf"], d["lt"]),
    "cash_lt_wr":        lambda d: _div(d["che"], d["lt"]),
    "cash_ratio_wr":     lambda d: _div(d["che"], d["lct"], positive=True),
    "cfm_wr":            lambda d: _div(d["ib"] + d["dp"], d["sale"]),
    "curr_debt_wr":      lambda d: _div(d["lct"], d["lt"]),
    "curr_ratio_wr":     lambda d: _div(d["act"], d["lct"]),
    "de_ratio_wr":       lambda d: _div(d["lt"], d["ceq"] + _z(d["pstk"])),
    "debt_assets_wr":    lambda d: _div(d["lt"], d["at"]),
    "debt_at_wr":        lambda d: _div(d["dltt"] + d["dlc"], d["at"]),
    "debt_capital_wr":   lambda d: _div(d["ap"] + _z(d["dlc"]) + _z(d["dltt"]),
                                        d["ap"] + _z(d["dlc"]) + _z(d["dltt"]) + d["ceq"] + _z(d["pstk"])),
    "debt_ebitda_wr":    lambda d: _div(d["dltt"] + d["dlc"], d["ebitda"]),
    "debt_invcap_wr":    lambda d: _div(d["dltt"], d["icapt"].where(d["icapt"] > 0)),
    "dltt_be_wr":        lambda d: _div(d["dltt"], d["be"], positive=True),
    "dpr_wr":            lambda d: _div(d["dv"], d["ibadj"]),
    "efftax_wr":         lambda d: _div(d["txt"], d["pi"]),
    "equity_invcap_wr":  lambda d: _div(d["ceq"], d["icapt"], positive=True),
    "fcf_ocf_wr":        lambda d: _div(d["ocf"] - d["capx"], d["ocf"], positive=True),
    "gpm_wr":            lambda d: _div(d["sale"] - d["cogs"], d["sale"]),
    "gprof_wr":          lambda d: _div(d["sale"] - d["cogs"], d["at"]),
    "int_debt_wr":       lambda d: _div(d["xint"], d["dltt"]),
    "int_totdebt_wr":    lambda d: _div(d["xint"], d["dltt"] + d["dlc"]),
    "intcov_ratio_wr":   lambda d: _div(d["ebit"], d["xint"]),
    "intcov_wr":         lambda d: _div(d["xint"] + d["ib"], d["xint"]),
    "invt_act_wr":       lambda d: _div(d["invt"], d["act"]),
    "invturn_wr":        lambda d: _div(d["cogs"], (d["invt"] + d["l_invt"]) / 2, positive=True),
    "lt_debt_wr":        lambda d: _div(d["dltt"], d["lt"]),
    "lt_ppent_wr":       lambda d: _div(d["lt"], d["ppent"]),
    "npm_wr":            lambda d: _div(d["ib"], d["sale"]),
    "ocf_lct_wr":        lambda d: _div(d["ocf"], d["lct"]),
    "opmad_wr":          lambda d: _div(d["ebit"], d["sale"]),
    "opmbd_wr":          lambda d: _div(d["ebitda"], d["sale"]),
    "pay_turn_wr":       lambda d: _div(d["cogs"] + d["d_invt"], (d["ap"] + d["l_ap"]) / 2, positive=True),
    "pretret_earnat_wr": lambda d: _div(d["ebitda"], d["ppent"] + d["act"]),
    "pretret_noa_wr":    lambda d: _div(d["ebit"], d["ppent"] + d["act"] - d["lct"]),
    "profit_lct_wr":     lambda d: _div(d["ebitda"], d["lct"]),
    "ptpm_wr":           lambda d: _div(d["ebit"] - d["xint"] + _z(d["spi"]) + _z(d["nopi"]), d["sale"]),
    "quick_ratio_wr":    lambda d: _div(d["act"] - d["invt"], d["lct"], positive=True),
    "rd_sale_wr":        lambda d: _div(_z(d["xrd"]), d["sale"]).clip(lower=0),
    "rect_act_wr":       lambda d: _div(d["rect"], d["act"]),
    "rect_turn_wr":      lambda d: _div(d["sale"], (d["rect"] + d["l_rect"]) / 2),
    "roa_wr":            lambda d: _div(d["ebitda"], d["at"]),
    "roce_wr":           lambda d: _div(d["ebit"], d["dltt"] + _z(d["pstk"]) + d["dlc"] + d["ceq"]),
    "roe_wr":            lambda d: _div(d["ib"], d["be"], positive=True),
    "short_debt_wr":     lambda d: _div(d["dlc"], d["dltt"] + d["dlc"]),
    "totdebt_invcap_wr": lambda d: _div(d["dltt"] + d["dlc"], d["icapt"], positive=True),
}

# ---------- 39 anomaly variables ----------
AN = {
    "assetgrowth_an":     lambda d: _div(d["d_at"], d["l_at"]),
    "assetturnover2_an":  lambda d: _div(d["sale"], (d["at"] + d["l_at"]) / 2),
    "assetturnover_an":   lambda d: _div(d["sale"], d["at"]),
    "bookequitygrowth_an":lambda d: _div(d["be"], d["l_be"]) - 1,
    "capexgrowth_an":     lambda d: _div(d["d_capx"], d["l_capx"]),
    "cash2assets_an":     lambda d: _div(d["che"], d["at"]),
    "cf2debt_an":         lambda d: _div(d["ib"] + d["dp"], (d["lt"] + d["l_lt"]) / 2),
    "chbe_an":            lambda d: _div(d["be"] - d["l_be"], d["l_at"]),
    "chca_an":            lambda d: _div(d["d_act"] - d["d_che"], d["l_at"]),
    "chceq_an":           lambda d: _div(d["ceq"] - d["l_ceq"], d["l_ceq"]),
    "chcl_an":            lambda d: _div(d["d_lct"] - _z(d["d_dlc"]), d["l_at"]),
    "chcurrentratio_an":  lambda d: _div(_div(d["act"], d["lct"]) - _div(d["l_act"], d["l_lct"]),
                                         _div(d["l_act"], d["l_lct"])),
    "chfnl_an":           lambda d: _div(_z(d["d_dltt"]) + _z(d["d_dlc"]) + _z(d["d_pstk"]), d["l_at"]),
    "chlt_an":            lambda d: _div(d["lt"], d["l_lt"]) - 1,
    "chnccl_an":          lambda d: _div(d["d_lt"] - d["d_lct"] - _z(d["d_dltt"]), d["l_at"]),
    "debt2tang_an":       lambda d: _div(d["che"] + 0.715 * d["rect"] + 0.547 * d["invt"]
                                         + 0.535 * d["ppent"], d["at"]),
    "deprn_an":           lambda d: _div(d["dp"], d["ppent"]),
    "ebitda2revenue_an":  lambda d: _div(d["ebitda"], d["sale"]),
    "grossmargin_an":     lambda d: _div(d["sale"] - d["cogs"], d["sale"]),
    "grossprofit_an":     lambda d: _div(d["sale"] - d["cogs"], d["at"]),
    "inventorychange_an": lambda d: _div(d["d_invt"], 0.5 * d["l_at"] + 0.5 * d["at"]),
    "inventorygrowth_an": lambda d: _div(d["d_invt"], d["l_invt"]),
    "investment_an":      lambda d: _div(d["d_at"], d["at"]),
    "liquid2assets_an":   lambda d: _div(d["che"] + 0.75 * (d["act"] - d["che"])
                                         + 0.5 * (d["at"] - d["act"]), d["l_at"]),
    "marginch_an":        lambda d: _div(d["ib"], d["sale"]) - _div(d["l_ib"], d["l_sale"]),
    "opleverage_an":      lambda d: _div(d["cogs"] + d["xsga"], d["at"]),
    "pchdeprn_an":        lambda d: _div(_div(d["dp"], d["ppent"]) - _div(d["l_dp"], d["l_ppent"]),
                                         _div(d["l_dp"], d["l_ppent"])),
    "pchgm2pchsale_an":   lambda d: (_div((d["sale"] - d["cogs"]) - (d["l_sale"] - d["l_cogs"]),
                                          d["l_sale"] - d["l_cogs"])
                                     - _div(d["sale"] - d["l_sale"], d["l_sale"])),
    "pchquickratio_an":   lambda d: _div(_div(d["act"] - d["invt"], d["lct"])
                                         - _div(d["l_act"] - d["l_invt"], d["l_lct"]),
                                         _div(d["l_act"] - d["l_invt"], d["l_lct"])),
    "pchsale2pchinvt_an": lambda d: _div(d["sale"] - d["l_sale"], d["l_sale"]) - _div(d["invt"] - d["l_invt"], d["l_invt"]),
    "pchsale2pchrect_an": lambda d: _div(d["sale"] - d["l_sale"], d["l_sale"]) - _div(d["rect"] - d["l_rect"], d["l_rect"]),
    "pchsale2pchxsga_an": lambda d: _div(d["sale"] - d["l_sale"], d["l_sale"]) - _div(d["xsga"] - d["l_xsga"], d["l_xsga"]),
    "pchsales2inv_an":    lambda d: _div(_div(d["sale"], d["invt"]) - _div(d["l_sale"], d["l_invt"]),
                                         _div(d["l_sale"], d["l_invt"])),
    "profitability_an":   lambda d: _div(d["sale"] - d["cogs"] - _z(d["xint"]) - d["xsga"], d["be"]),
    "roic_an":            lambda d: _div(d["ebit"] - _z(d["nopi"]), d["ceq"] + d["lt"] - d["che"]),
    "sales2cash_an":      lambda d: _div(d["sale"], d["che"]),
    "sales2inv_an":       lambda d: _div(d["sale"], d["invt"]),
    "sales2rec_an":       lambda d: _div(d["sale"], d["rect"]),
    "salesgrowth_an":     lambda d: _div(d["sale"], d["l_sale"]) - 1,
}

# ---------- 4 size variables ----------
SIZE = {
    "assets":          lambda d: d["at"],
    "book":            lambda d: d["be"],
    "sale_ac":         lambda d: d["sale"],
    "debt_book_value": lambda d: _z(d["dltt"]) + _z(d["dlc"]) + _z(d["pstk"]),
}

FEATURES = list(WR) + list(AN) + list(SIZE) + ["CAPMbeta", "industry"]   # 97 in total
assert len(FEATURES) == 97, len(FEATURES)

# 11 categories: as reported in the paper (Tables 7-9) where available; otherwise following the WRDS ratio-suite classification
CATEGORY = {}
for v in ["roe_wr", "profitability_an", "aftret_invcapx_wr", "pretret_noa_wr", "roic_an",
          "roce_wr", "pretret_earnat_wr", "roa_wr", "gprof_wr", "opmad_wr", "npm_wr",
          "gpm_wr", "ebitda2revenue_an", "aftret_eq_wr", "aftret_equity_wr", "opmbd_wr",
          "ptpm_wr", "cfm_wr", "efftax_wr", "grossmargin_an", "grossprofit_an", "marginch_an"]:
    CATEGORY[v] = "Profitability"
for v in ["rd_sale_wr", "salesgrowth_an", "chbe_an", "assetgrowth_an", "bookequitygrowth_an",
          "capexgrowth_an", "chceq_an", "investment_an", "inventorygrowth_an", "pchgm2pchsale_an"]:
    CATEGORY[v] = "Growth"
for v in ["opleverage_an", "at_turn_wr", "sales2rec_an", "rect_turn_wr", "sales2cash_an",
          "assetturnover_an", "assetturnover2_an", "invturn_wr", "pay_turn_wr", "sales2inv_an",
          "deprn_an", "pchdeprn_an", "pchsale2pchinvt_an", "pchsale2pchrect_an",
          "pchsale2pchxsga_an", "pchsales2inv_an", "inventorychange_an"]:
    CATEGORY[v] = "Efficiency"
for v in ["lt_ppent_wr", "Accrual_wr", "lt_debt_wr", "dltt_be_wr", "cash_lt_wr", "invt_act_wr",
          "rect_act_wr", "debt_ebitda_wr", "short_debt_wr", "curr_debt_wr", "profit_lct_wr",
          "ocf_lct_wr", "cash_debt_wr", "fcf_ocf_wr", "chca_an", "chcl_an", "chnccl_an",
          "chfnl_an", "cf2debt_an"]:
    CATEGORY[v] = "Financial soundness"
for v in ["debt_at_wr", "debt_assets_wr", "debt_capital_wr", "de_ratio_wr", "intcov_wr",
          "intcov_ratio_wr", "int_debt_wr", "int_totdebt_wr", "chlt_an", "debt2tang_an"]:
    CATEGORY[v] = "Solvency"
for v in ["cash_ratio_wr", "quick_ratio_wr", "curr_ratio_wr", "cash_conversion_wr",
          "cash2assets_an", "liquid2assets_an", "chcurrentratio_an", "pchquickratio_an"]:
    CATEGORY[v] = "Liquidity"
for v in ["capital_ratio_wr", "equity_invcap_wr", "debt_invcap_wr", "totdebt_invcap_wr"]:
    CATEGORY[v] = "Capitalization"
CATEGORY["dpr_wr"] = "Payout"
for v in SIZE:
    CATEGORY[v] = "Size"
CATEGORY["CAPMbeta"] = "CAPMbeta"
CATEGORY["industry"] = "Industry"
assert set(CATEGORY) == set(FEATURES)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Input: merged monthly panel (base accounting variables, CAPMbeta, ff49).
    Output: DataFrame with the same rows and the 97 FEATURES columns; infinities set to NaN."""
    d = panel
    out = {}
    for name, f in {**WR, **AN, **SIZE}.items():
        out[name] = f(d).astype(float)
    X = pd.DataFrame(out, index=d.index).replace([np.inf, -np.inf], np.nan)
    X["CAPMbeta"] = d["CAPMbeta"].astype(float)
    X["industry"] = pd.Categorical(d["ff49"], categories=list(range(1, 50)))
    return X[FEATURES]
