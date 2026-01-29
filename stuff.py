df2 = add_transformer_metrics(df, spec)

cols = [
    "Transformer Serial Number",
    "Transformer Design Number",
    "kVA_num",
    "HV_VLL_num",
    "LV_VLL_num",
    "Z_percent_HV_%",
    "LoadLoss_corr_W",
    "CoreLoss_W",
    "HV_R_unbalance_%",
    "LV_R_unbalance_%",
    "Exc_I_unbalance_%",
    "FAIL_FLAGS",
]
display(df2[cols].sort_values("FAIL_FLAGS"))