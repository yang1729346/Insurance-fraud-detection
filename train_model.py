"""
train_model.py
运行此脚本，用训练数据生成 model_artifacts.pkl 和 meta_info.json。

用法：
    python train_model.py --train path/to/train.csv
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─── 配置 ─────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_FOLDS       = 5
HIGH_CARD_COLS = ["auto_model", "insured_hobbies", "insured_occupation", "auto_make"]
LOW_CARD_COLS  = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_relationship", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city",
    "property_damage", "police_report_available",
]
DROP_COLS      = ["policy_id", "policy_bind_date", "incident_date", "policy_end_date", "insured_zip"]
HIGH_RISK_OCC  = {"handlers-cleaners", "machine-op-inspct", "transport-moving", "craft-repair", "farming-fishing"}
HIGH_RISK_HOB  = {"skydiving", "bungie-jumping", "base-jumping", "chess", "cross-fit", "paintball", "board-games"}


# ─── Target Encoder ───────────────────────────────────────────────────────────
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self._stats: dict = {}
        self._global_mean: float = 0.0

    def fit(self, X, y):
        self._global_mean = float(y.mean())
        df = X.copy()
        df["__t__"] = y.values
        for col in self.cols:
            agg = df.groupby(df[col].astype(str))["__t__"].agg(["mean", "count"])
            n, m = agg["count"], agg["mean"]
            self._stats[col] = (
                (n * m + self.smoothing * self._global_mean) / (n + self.smoothing)
            ).to_dict()
        return self

    def transform(self, X):
        out = X.copy()
        for col in self.cols:
            out[col] = (
                out[col].astype(str).map(self._stats[col]).fillna(self._global_mean).astype(float)
            )
        return out


# ─── 特征工程 ─────────────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    df["incident_date"]    = pd.to_datetime(df["incident_date"],    errors="coerce")
    df["policy_end_date"]  = df["policy_bind_date"] + pd.DateOffset(years=1)

    for c in ["collision_type", "property_damage", "police_report_available"]:
        if c in df.columns:
            df[c] = df[c].replace("?", "Unknown")
    if "authorities_contacted" in df.columns:
        df["authorities_contacted"] = df["authorities_contacted"].fillna("Unknown")

    df["days_since_policy_start"] = (df["incident_date"] - df["policy_bind_date"]).dt.days.astype(float)
    df["days_to_policy_end"]      = (df["policy_end_date"] - df["incident_date"]).dt.days.astype(float)
    df["is_night_incident"]       = df["incident_hour_of_the_day"].isin([0,1,2,3,4,5,22,23]).astype(int)
    df["is_weekend_incident"]     = (df["incident_date"].dt.dayofweek >= 5).astype(int)
    df["is_early_claim"]          = (df["days_since_policy_start"] < 30).astype(int)
    df["is_near_expiry_claim"]    = (df["days_to_policy_end"] < 30).astype(int)

    total = df["total_claim_amount"].replace(0, np.nan)
    df["injury_to_total_ratio"]   = df["injury_claim"]   / total
    df["vehicle_to_total_ratio"]  = df["vehicle_claim"]  / total
    df["claim_to_premium_ratio"]  = df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)
    df["vehicle_age"]             = df["incident_date"].dt.year - df["auto_year"]
    df["is_near_deductible"]      = ((df["total_claim_amount"] - df["policy_deductable"]).abs() < 1000).astype(int)
    df["claim_over_deductable_x"] = df["total_claim_amount"] / df["policy_deductable"].replace(0, np.nan)

    def parse_csl(v):
        try: return int(str(v).split("/")[1])
        except: return 300

    df["csl_max_limit"]           = df["policy_csl"].apply(parse_csl)
    df["claim_to_csl_ratio"]      = df["total_claim_amount"] / (df["csl_max_limit"] * 1000)
    df["no_witness"]              = (df["witnesses"] == 0).astype(int)
    df["no_police_report"]        = df["police_report_available"].isin(["NO", "Unknown"]).astype(int)
    df["no_property_damage_rec"]  = df["property_damage"].isin(["NO", "Unknown"]).astype(int)
    df["evidence_gap_score"]      = df["no_witness"] + df["no_police_report"] + df["no_property_damage_rec"]
    df["is_high_risk_occ"]        = df["insured_occupation"].isin(HIGH_RISK_OCC).astype(int)
    df["is_high_risk_hobby"]      = df["insured_hobbies"].isin(HIGH_RISK_HOB).astype(int)
    df["high_risk_profile"]       = (df["is_high_risk_occ"] & df["is_high_risk_hobby"]).astype(int)
    df["is_extreme_hobby"]        = df["insured_hobbies"].isin(
        {"skydiving", "bungie-jumping", "base-jumping", "chess", "cross-fit"}
    ).astype(int)
    df["severity_type_combo"]     = (
        df["incident_severity"].str.replace(" ", "_") + "__" +
        df["incident_type"].str.replace(" ", "_")
    )
    df["net_capital"]             = df["capital-gains"] + df["capital-loss"]
    df["night_no_witness"]        = (df["is_night_incident"] & df["no_witness"]).astype(int)
    df["collision_type_missing"]  = (df["collision_type"] == "Unknown").astype(int)
    return df


# ─── 主训练流程 ───────────────────────────────────────────────────────────────
def train(train_csv: str, output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 加载数据: {train_csv}")
    train_df = pd.read_csv(train_csv)
    y = train_df["fraud"]
    X = engineer(train_df.drop(columns=["fraud"]))
    for c in DROP_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])

    ALL_OHE = ["severity_type_combo"] + [c for c in LOW_CARD_COLS if c in X.columns]
    ALL_TE  = [c for c in HIGH_CARD_COLS if c in X.columns]

    print(f"[2/5] 特征工程完成：{X.shape[1]} 列")

    # 编码
    te  = TargetEncoder(cols=ALL_TE, smoothing=10.0)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)

    X_te   = te.fit(X[ALL_TE], y).transform(X[ALL_TE])
    ohe.fit(X[ALL_OHE].astype(str))
    ohe_arr = ohe.transform(X[ALL_OHE].astype(str))
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(ALL_OHE), index=X.index)
    num_cols = [c for c in X.select_dtypes(include=np.number).columns if c not in ALL_OHE and c not in ALL_TE]
    X_enc   = pd.concat([X[num_cols], X_te, ohe_df], axis=1)

    print(f"[3/5] 编码完成：{X_enc.shape[1]} 列（含 OHE 展开）")

    # OOF 训练
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    hgb_oof = np.zeros(len(y))
    rf_oof  = np.zeros(len(y))

    print("[4/5] OOF 交叉验证…")
    for fold, (tr_i, va_i) in enumerate(cv.split(X_enc, y), 1):
        sw = compute_sample_weight("balanced", y.iloc[tr_i])
        mh = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            min_samples_leaf=15, l2_regularization=0.1,
            early_stopping=False, random_state=RANDOM_STATE,
        )
        mr = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        mh.fit(X_enc.iloc[tr_i], y.iloc[tr_i], sample_weight=sw)
        mr.fit(X_enc.iloc[tr_i], y.iloc[tr_i])
        hgb_oof[va_i] = mh.predict_proba(X_enc.iloc[va_i])[:, 1]
        rf_oof[va_i]  = mr.predict_proba(X_enc.iloc[va_i])[:, 1]
        fold_auc = roc_auc_score(y.iloc[va_i], hgb_oof[va_i])
        print(f"  Fold {fold}: HGB AUC={fold_auc:.4f}")

    meta_lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    meta_lr.fit(np.column_stack([hgb_oof, rf_oof]), y)
    blend = meta_lr.predict_proba(np.column_stack([hgb_oof, rf_oof]))[:, 1]

    oof_auc   = roc_auc_score(y, hgb_oof)
    blend_auc = roc_auc_score(y, blend)
    blend_f1  = f1_score(y, (blend >= 0.5).astype(int))
    print(f"  OOF AUC={oof_auc:.4f}  Blend AUC={blend_auc:.4f}  F1={blend_f1:.4f}")

    # 全集重训练
    print("[5/5] 全集重训练…")
    sw_full = compute_sample_weight("balanced", y)
    hgb_final = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=15, l2_regularization=0.1,
        early_stopping=False, random_state=RANDOM_STATE,
    )
    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    hgb_final.fit(X_enc, y, sample_weight=sw_full)
    rf_final.fit(X_enc, y)

    feat_imp = pd.Series(rf_final.feature_importances_, index=X_enc.columns).sort_values(ascending=False)

    artifacts = {
        "te": te, "ohe": ohe, "hgb": hgb_final, "rf": rf_final, "meta": meta_lr,
        "feature_names": list(X_enc.columns),
        "num_cols": num_cols, "ohe_cols": ALL_OHE, "te_cols": ALL_TE,
        "metrics": {
            "oof_auc":   round(oof_auc,   4),
            "blend_auc": round(blend_auc, 4),
            "blend_f1":  round(blend_f1,  4),
        },
        "feat_imp":    feat_imp.head(20).round(4).to_dict(),
        "fraud_rate":  float(y.mean()),
        "train_size":  int(len(y)),
    }
    pkl_path = out / "model_artifacts.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)

    meta_info = {
        "metrics":         artifacts["metrics"],
        "fraud_rate":      artifacts["fraud_rate"],
        "train_size":      artifacts["train_size"],
        "top_features":    list(feat_imp.head(10).index),
        "feat_imp_values": feat_imp.head(10).round(4).to_dict(),
    }
    json_path = out / "meta_info.json"
    with open(json_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"\n✅ 完成！")
    print(f"   模型文件  → {pkl_path}")
    print(f"   摘要 JSON → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv", help="训练集 CSV 路径")
    parser.add_argument("--out",   default=".",          help="输出目录")
    args = parser.parse_args()
    train(args.train, args.out)
