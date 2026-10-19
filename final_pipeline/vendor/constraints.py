"""
Domain constraints auditor and enforcer for NSL-KDD and UNSW-NB15 in Experiment 2.
Directly addresses physical network identities and fixes the tcprtt inflation bug.
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def audit_constraints(df: pd.DataFrame, dataset_name: str) -> Dict[str, int]:
    """Audit the number of constraint violations in a dataframe."""
    violations = {}
    name = dataset_name.lower()
    
    if "nsl" in name:
        # Rate features in [0, 1]
        rate_cols = [c for c in df.columns if c.endswith("_rate")]
        for c in rate_cols:
            if c in df.columns:
                violations[f"{c}_out_of_[0,1]"] = int(((df[c] < 0) | (df[c] > 1)).sum())
                
        # Non-negative counts/bytes
        non_neg = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
                    "count", "srv_count", "dst_host_count", "dst_host_srv_count"]
        for c in non_neg:
            if c in df.columns:
                violations[f"{c}_negative"] = int((df[c] < 0).sum())
                
        # Logical rule: if logged_in == 0, root_shell and su_attempted should be 0
        if "logged_in" in df.columns:
            not_logged = df["logged_in"] == 0
            if "root_shell" in df.columns:
                violations["root_shell_without_login"] = int((not_logged & (df["root_shell"] > 0)).sum())
            if "su_attempted" in df.columns:
                violations["su_attempted_without_login"] = int((not_logged & (df["su_attempted"] > 0)).sum())
                
    elif "unsw" in name:
        # Physical non-negativity
        non_neg = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
                    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"]
        for c in non_neg:
            if c in df.columns:
                violations[f"{c}_negative"] = int((df[c] < 0).sum())
                
        # TTL in [0, 255]
        for c in ["sttl", "dttl"]:
            if c in df.columns:
                violations[f"{c}_out_of_[0,255]"] = int(((df[c] < 0) | (df[c] > 255)).sum())
                
        # tcprtt == synack + ackdat exact identity
        if all(c in df.columns for c in ["tcprtt", "synack", "ackdat"]):
            diff = np.abs(df["tcprtt"] - (df["synack"] + df["ackdat"]))
            violations["tcprtt_identity_mismatch"] = int((diff > 1e-4).sum())
            
        # Non-TCP zeroing: swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat == 0 when proto != 'tcp'
        if "proto" in df.columns:
            non_tcp = df["proto"].astype(str).str.lower() != "tcp"
            tcp_fields = ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"]
            for c in tcp_fields:
                if c in df.columns:
                    violations[f"{c}_nonzero_for_non_tcp"] = int((non_tcp & (df[c] > 0)).sum())
                    
    return {k: v for k, v in violations.items() if v > 0}


def enforce_nslkdd_constraints(df: pd.DataFrame, train_normal: pd.DataFrame = None) -> pd.DataFrame:
    """Enforce physical and logical constraints on NSL-KDD synthetic records."""
    df = df.copy()
    
    # 1. Rate columns strictly clipped to [0, 1]
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in rate_cols:
        df[c] = df[c].clip(0.0, 1.0)
        
    # 2. Non-negative counts and durations
    non_neg = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
                "count", "srv_count", "dst_host_count", "dst_host_srv_count",
                "hot", "num_failed_logins", "num_compromised", "num_root",
                "num_file_creations", "num_shells", "num_access_files"]
    for c in non_neg:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
            if c not in ["duration", "src_bytes", "dst_bytes"]:
                df[c] = np.rint(df[c])
                
    # 3. Binary columns snapped to {0, 1}
    binary_cols = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
    for c in binary_cols:
        if c in df.columns:
            df[c] = np.where(df[c] >= 0.5, 1, 0)
            
    # 4. Logical dependency: no root_shell or su_attempted if not logged in
    if "logged_in" in df.columns:
        not_logged = df["logged_in"] == 0
        if "root_shell" in df.columns:
            df.loc[not_logged, "root_shell"] = 0
        if "su_attempted" in df.columns:
            df.loc[not_logged, "su_attempted"] = 0
            
    # 5. Constant zero columns
    for c in ["num_outbound_cmds", "is_host_login"]:
        if c in df.columns:
            df[c] = 0
            
    return df


def enforce_unsw_constraints(df: pd.DataFrame, max_tcprtt: float = 3.18) -> pd.DataFrame:
    """
    Enforce physical and logical constraints on UNSW-NB15 synthetic records.
    Directly fixes the tcprtt inflation bug and non-TCP zeroing.
    """
    df = df.copy()
    
    # 1. Non-negativity for physical metrics
    non_neg = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
                "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"]
    for c in non_neg:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0)
            
    # 2. Integer packet and loss counts
    int_cols = ["spkts", "dpkts", "sloss", "dloss", "trans_depth", "ct_state_ttl",
                "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]
    for c in int_cols:
        if c in df.columns:
            df[c] = np.rint(df[c]).clip(lower=0)
            
    # 3. TTL strictly in [0, 255]
    for c in ["sttl", "dttl"]:
        if c in df.columns:
            df[c] = np.rint(df[c]).clip(0, 255)
            
    # 4. Binary indicators
    for c in ["is_sm_ips_ports", "is_ftp_login"]:
        if c in df.columns:
            df[c] = np.where(df[c] >= 0.5, 1, 0)
            
    # 5. Non-TCP protocol handling
    if "proto" in df.columns:
        is_tcp = df["proto"].astype(str).str.lower() == "tcp"
        tcp_fields = ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"]
        for c in tcp_fields:
            if c in df.columns:
                df.loc[~is_tcp, c] = 0.0
                
    # 6. EXACT IDENTITY FIX: tcprtt = synack + ackdat with upper bound safety
    if all(c in df.columns for c in ["tcprtt", "synack", "ackdat"]):
        # Recompute sum
        raw_sum = df["synack"] + df["ackdat"]
        
        # If raw_sum exceeds max allowable training tcprtt, proportionately scale down synack & ackdat
        excess_mask = raw_sum > max_tcprtt
        if excess_mask.any():
            scale = max_tcprtt / np.maximum(raw_sum, 1e-9)
            df.loc[excess_mask, "synack"] = df.loc[excess_mask, "synack"] * scale[excess_mask]
            df.loc[excess_mask, "ackdat"] = df.loc[excess_mask, "ackdat"] * scale[excess_mask]
            
        df["tcprtt"] = df["synack"] + df["ackdat"]
        
    return df

