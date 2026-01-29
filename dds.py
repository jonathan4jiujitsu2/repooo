import numpy as np
import pandas as pd
import math
import re

# ----------------------------
# Utilities
# ----------------------------
def to_float(x):
    """Convert strings/numbers like '1,870', ' 12.84 ', None -> float."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan

def pct_unbalance(vals):
    """max deviation from mean / mean * 100"""
    v = np.array([to_float(x) for x in vals], dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan
    mu = v.mean()
    if mu == 0:
        return np.nan
    return float(np.max(np.abs(v - mu)) / mu * 100)

def rated_current_3ph(kva, v_ll):
    """I = kVA*1000 / (sqrt(3)*V_LL)"""
    if not np.isfinite(kva) or not np.isfinite(v_ll) or v_ll == 0:
        return np.nan
    return (kva * 1000.0) / (math.sqrt(3) * v_ll)

def convert_to_vll(v_reported, mode_text):
    """
    If mode contains 'PH-N' or 'A-N'/'B-N'/'C-N' -> treat as phase-to-neutral and convert to LL using sqrt(3).
    Otherwise assume already line-line.
    """
    v = to_float(v_reported)
    if not np.isfinite(v):
        return np.nan

    mode = (mode_text or "").upper()

    # Your column label says "A-N or A-B" etc; the mode field usually says "Ph-N" or similar
    if ("PH-N" in mode) or (("PH" in mode) and ("N" in mode)) or ("A-N" in mode) or ("B-N" in mode) or ("C-N" in mode):
        return float(math.sqrt(3) * v)
    return float(v)

# ----------------------------
# Main computation
# ----------------------------
def add_transformer_metrics(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    df: your raw dataframe (one row per transformer)
    spec: dictionary of thresholds/limits

    Returns df with new computed columns + a FAIL_FLAGS column.
    """
    out = df.copy()

    # Nameplate basics
    out["kVA_num"] = out["KVA"].map(to_float)
    out["HV_VLL_num"] = out["HV Rate Phase to Phase Voltage"].map(to_float)
    out["LV_VLL_num"] = out["LV Rate Phase to Phase Voltage"].map(to_float)

    # Rated currents
    out["I_rated_HV_A"] = out.apply(lambda r: rated_current_3ph(r["kVA_num"], r["HV_VLL_num"]), axis=1)
    out["I_rated_LV_A"] = out.apply(lambda r: rated_current_3ph(r["kVA_num"], r["LV_VLL_num"]), axis=1)

    # ----------------------------
    # DC resistance unbalance
    # ----------------------------
    out["HV_R_unbalance_%"] = out.apply(
        lambda r: pct_unbalance([r.get("H1-H2"), r.get("H2-H3"), r.get("H3-H1")]), axis=1
    )
    out["LV_R_unbalance_%"] = out.apply(
        lambda r: pct_unbalance([r.get("X1-X2"), r.get("X2-X3"), r.get("X3-X1")]), axis=1
    )

    # ----------------------------
    # Impedance test: currents, volts, watts
    # ----------------------------
    out["Imp_Ia"] = out["Ampere Reading Phase A"].map(to_float)
    out["Imp_Ib"] = out["Ampere Reading Phase B"].map(to_float)
    out["Imp_Ic"] = out["Ampere Reading Phase C"].map(to_float)

    out["Imp_I_avg_A"] = out[["Imp_Ia","Imp_Ib","Imp_Ic"]].mean(axis=1, skipna=True)
    out["Imp_I_unbalance_%"] = out.apply(lambda r: pct_unbalance([r["Imp_Ia"], r["Imp_Ib"], r["Imp_Ic"]]), axis=1)

    # Voltages: the mode column tells Ph-N vs line-line
    mode_col = "Impedance Voltage Measurement"
    out["Imp_Va_rep"] = out["Voltmeter Reading, Pha A-N or A-B, Volt"].map(to_float)
    out["Imp_Vb_rep"] = out["Voltmeter Reading, Pha B-N or B-C, Volt"].map(to_float)
    out["Imp_Vc_rep"] = out["Voltmeter Reading, Pha C-N or C-A, Volt"].map(to_float)

    out["Imp_V_avg_reported_V"] = out[["Imp_Va_rep","Imp_Vb_rep","Imp_Vc_rep"]].mean(axis=1, skipna=True)

    out["Imp_V_sc_LL_V"] = out.apply(lambda r: convert_to_vll(r["Imp_V_avg_reported_V"], r.get(mode_col)), axis=1)

    # %Z on HV base (typical)
    out["Z_percent_HV_%"] = (out["Imp_V_sc_LL_V"] / out["HV_VLL_num"]) * 100.0

    # Load loss (sum 3 wattmeters)
    out["Imp_W1"] = out["Wattmeter #1 reading Watt ( + or - )"].map(to_float)
    out["Imp_W2"] = out["Wattmeter #2 reading Watt ( + or - )"].map(to_float)
    out["Imp_W3"] = out["Wattmeter #3 reading Watt ( + or - )"].map(to_float)
    out["LoadLoss_meas_W"] = out[["Imp_W1","Imp_W2","Imp_W3"]].sum(axis=1, skipna=True)

    # Correct load loss to rated current (use HV rated current unless you set spec["loss_side"]="LV")
    loss_side = spec.get("loss_side", "HV").upper()
    out["I_rated_for_loss_A"] = np.where(loss_side=="LV", out["I_rated_LV_A"], out["I_rated_HV_A"])

    out["LoadLoss_corr_W"] = out.apply(
        lambda r: (r["LoadLoss_meas_W"] * (r["I_rated_for_loss_A"]/r["Imp_I_avg_A"])**2)
        if (np.isfinite(r["LoadLoss_meas_W"]) and np.isfinite(r["I_rated_for_loss_A"]) and np.isfinite(r["Imp_I_avg_A"]) and r["Imp_I_avg_A"]>0)
        else np.nan,
        axis=1
    )

    out["Imp_negative_wattmeter_flag"] = out.apply(
        lambda r: any([(x < 0) for x in [r["Imp_W1"], r["Imp_W2"], r["Imp_W3"]] if np.isfinite(x)]),
        axis=1
    )

    # ----------------------------
    # Core loss test
    # ----------------------------
    out["Exc_Ia"] = out["Phase A"].map(to_float)
    out["Exc_Ib"] = out["Phase B"].map(to_float)
    out["Exc_Ic"] = out["Phase C"].map(to_float)

    out["Exc_I_unbalance_%"] = out.apply(lambda r: pct_unbalance([r["Exc_Ia"], r["Exc_Ib"], r["Exc_Ic"]]), axis=1)

    out["Core_W1"] = out["Wattmeter #1"].map(to_float)
    out["Core_W2"] = out["Wattmeter #2"].map(to_float)
    out["Core_W3"] = out["Wattmeter #3"].map(to_float)
    out["CoreLoss_W"] = out[["Core_W1","Core_W2","Core_W3"]].sum(axis=1, skipna=True)

    out["Core_negative_wattmeter_flag"] = out.apply(
        lambda r: any([(x < 0) for x in [r["Core_W1"], r["Core_W2"], r["Core_W3"]] if np.isfinite(x)]),
        axis=1
    )

    # ----------------------------
    # Flagging logic
    # ----------------------------
    def flag_row(r):
        flags = []

        # %Z limits
        z = r.get("Z_percent_HV_%", np.nan)
        if np.isfinite(z):
            zmin = spec.get("z_percent_min")
            zmax = spec.get("z_percent_max")
            if zmin is not None and z < zmin:
                flags.append(f"%Z LOW ({z:.2f}% < {zmin}%)")
            if zmax is not None and z > zmax:
                flags.append(f"%Z HIGH ({z:.2f}% > {zmax}%)")

        # Load loss max
        pll = r.get("LoadLoss_corr_W", np.nan)
        pll_max = spec.get("load_loss_max_W")
        if pll_max is not None and np.isfinite(pll) and pll > pll_max:
            flags.append(f"LOAD LOSS HIGH ({pll:.0f} W > {pll_max} W)")

        # Core loss max
        pcl = r.get("CoreLoss_W", np.nan)
        pcl_max = spec.get("core_loss_max_W")
        if pcl_max is not None and np.isfinite(pcl) and pcl > pcl_max:
            flags.append(f"CORE LOSS HIGH ({pcl:.0f} W > {pcl_max} W)")

        # Excitation unbalance
        eunb = r.get("Exc_I_unbalance_%", np.nan)
        eunb_max = spec.get("exc_unbalance_max_pct")
        if eunb_max is not None and np.isfinite(eunb) and eunb > eunb_max:
            flags.append(f"EXC I UNBAL ({eunb:.1f}% > {eunb_max}%)")

        # Resistance unbalance
        runb_max = spec.get("resistance_unbalance_max_pct")
        if runb_max is not None:
            hv_runb = r.get("HV_R_unbalance_%", np.nan)
            lv_runb = r.get("LV_R_unbalance_%", np.nan)
            if np.isfinite(hv_runb) and hv_runb > runb_max:
                flags.append(f"HV R UNBAL ({hv_runb:.2f}% > {runb_max}%)")
            if np.isfinite(lv_runb) and lv_runb > runb_max:
                flags.append(f"LV R UNBAL ({lv_runb:.2f}% > {runb_max}%)")

        # Negative wattmeters (often wiring/polarity/recipe)
        if r.get("Imp_negative_wattmeter_flag", False):
            flags.append("NEG WATT (impedance section) — check CT/PT polarity/wiring")
        if r.get("Core_negative_wattmeter_flag", False):
            flags.append("NEG WATT (core loss) — check CT/PT polarity/wiring")

        return " | ".join(flags) if flags else "PASS (no flags)"

    out["FAIL_FLAGS"] = out.apply(flag_row, axis=1)

    return out

# ----------------------------
# Example spec dictionary (EDIT THESE)
# ----------------------------
spec = {
    # Impedance percent limits (example)
    "z_percent_min": 5.75,
    "z_percent_max": 6.25,

    # Loss limits (example)
    "load_loss_max_W": 5200,
    "core_loss_max_W": 600,

    # Balance limits (example)
    "exc_unbalance_max_pct": 10.0,
    "resistance_unbalance_max_pct": 2.0,

    # Which side the short-circuit/load-loss current is based on ("HV" or "LV")
    "loss_side": "HV"
}

# Usage:
# df2 = add_transformer_metrics(df, spec)
# df2[["Transformer Serial Number", "Z_percent_HV_%", "LoadLoss_corr_W", "CoreLoss_W", "FAIL_FLAGS"]].head()