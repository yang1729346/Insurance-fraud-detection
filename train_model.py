"""
train_model.py — 车险欺诈检测模型训练脚本
作者：杨磊磊

模型架构：XGBoost + HistGradientBoosting + RandomForest → OOF Stacking → LogisticRegression 元学习器

用法：
    pip install xgboost scikit-learn pandas numpy
    python train_model.py --train train.csv --out .

输出：
    model_artifacts.pkl   — 供 app.py 加载的模型文件
    meta_info.json        — 模型摘要（指标、特征重要性），提交到 Git
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠  XGBoost 未安装，将跳过 XGB 模型。运行: pip install xgboost")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
N_FOLDS      = 5

HIGH_CARD_COLS = ["auto_model", "insured_hobbies", "insured_occupation", "auto_make"]
LOW_CARD_COLS  = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_relationship", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city",
    "property_damage", "police_report_available",
]
DROP_COLS     = ["policy_id", "policy_bind_date", "incident_date", "policy_end_date", "insured_zip"]
HIGH_RISK_OCC = {"handlers-cleaners", "machine-op-inspct", "transport-moving", "craft-repair", "farming-fishing"}
HIGH_RISK_HOB = {"skydiving", "bungie-jumping", "base-jumping", "chess", "cross-fit", "paintball", "board-games"}


# ══════════════════════════════════════════════════════════════════════════════
# Target Encoder（贝叶斯平滑）
# ══════════════════════════════════════════════════════════════════════════════
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    贝叶斯平滑 Target Encoding。
    公式：encoded = (n_i × mean_i + m × global_mean) / (n_i + m)
    平滑系数 m 越大，稀有类别越趋向全局均值，防止过拟合。
    """
    def __init__(self, cols: list, smoothing: float = 10.0):
        self.cols      = cols
        self.smoothing = smoothing
        self._stats: dict = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cols:
            out[col] = (
                out[col].astype(str)
                .map(self._stats[col])
                .fillna(self._global_mean)
                .astype(float)
            )
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 特征工程
# ══════════════════════════════════════════════════════════════════════════════
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """基于车险欺诈业务逻辑构造衍生特征，覆盖五个风险维度。"""
    df = df.copy()
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    df["incident_date"]    = pd.to_datetime(df["incident_date"],    errors="coerce")
    df["policy_end_date"]  = df["policy_bind_date"] + pd.DateOffset(years=1)
    for c in ["collision_type", "property_damage", "police_report_available"]:
        if c in df.columns: df[c] = df[c].replace("?", "Unknown")
    if "authorities_contacted" in df.columns:
        df["authorities_contacted"] = df["authorities_contacted"].fillna("Unknown")
    df["days_since_policy_start"] = (df["incident_date"] - df["policy_bind_date"]).dt.days.astype(float)
    df["days_to_policy_end"]      = (df["policy_end_date"] - df["incident_date"]).dt.days.astype(float)
    df["is_night_incident"]       = df["incident_hour_of_the_day"].isin([0,1,2,3,4,5,22,23]).astype(int)
    df["is_weekend_incident"]     = (df["incident_date"].dt.dayofweek >= 5).astype(int)
    df["is_early_claim"]          = (df["days_since_policy_start"] < 30).astype(int)
    df["is_near_expiry_claim"]    = (df["days_to_policy_end"] < 30).astype(int)
    total = df["total_claim_amount"].replace(0, np.nan)
    premium = df["policy_annual_premium"].replace(0, np.nan)
    df["injury_to_total_ratio"]   = df["injury_claim"] / total
    df["vehicle_to_total_ratio"]  = df["vehicle_claim"] / total
    df["claim_to_premium_ratio"]  = df["total_claim_amount"] / premium
    df["vehicle_age"]             = df["incident_date"].dt.year - df["auto_year"]
    df["is_near_deductible"]      = ((df["total_claim_amount"] - df["policy_deductable"]).abs() < 1000).astype(int)
    df["claim_over_deductable_x"] = df["total_claim_amount"] / df["policy_deductable"].replace(0, np.nan)
    def parse_csl(v):
        try: return int(str(v).split("/")[1])
        except: return 300
    df["csl_max_limit"]           = df["policy_csl"].apply(parse_csl)
    df["claim_to_csl_ratio"]      = df["total_claim_amount"] / (df["csl_max_limit"] * 1000)
    df["deductable_to_premium"]   = df["policy_deductable"] / premium
    df["no_witness"]              = (df["witnesses"] == 0).astype(int)
    df["no_police_report"]        = df["police_report_available"].isin(["NO", "Unknown"]).astype(int)
    df["no_property_damage_rec"]  = df["property_damage"].isin(["NO", "Unknown"]).astype(int)
    df["evidence_gap_score"]      = df["no_witness"] + df["no_police_report"] + df["no_property_damage_rec"]
    df["is_high_risk_occ"]        = df["insured_occupation"].isin(HIGH_RISK_OCC).astype(int)
    df["is_high_risk_hobby"]      = df["insured_hobbies"].isin(HIGH_RISK_HOB).astype(int)
    df["high_risk_profile"]       = (df["is_high_risk_occ"] & df["is_high_risk_hobby"]).astype(int)
    df["is_extreme_hobby"]        = df["insured_hobbies"].isin(
        {"skydiving", "bungie-jumping", "base-jumping", "chess", "cross-fit"}).astype(int)
    df["severity_type_combo"]     = (
        df["incident_severity"].str.replace(" ", "_") + "__" +
        df["incident_type"].str.replace(" ", "_"))
    df["net_capital"]             = df["capital-gains"] + df["capital-loss"]
    df["night_no_witness"]        = (df["is_night_incident"] & df["no_witness"]).astype(int)
    df["collision_type_missing"]  = (df["collision_type"] == "Unknown").astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════════════════════
def build_xgb(scale_pos_weight: float = 1.0):
    """XGBoost — tree_method=hist，L1+L2正则，scale_pos_weight 处理不平衡。"""
    return xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist", eval_metric="auc",
        use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1,
    )

def build_hgb():
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=15, max_leaf_nodes=63, l2_regularization=0.1,
        early_stopping=False, random_state=RANDOM_STATE,
    )

def build_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OOF Stacking
# ══════════════════════════════════════════════════════════════════════════════
def oof_stacking(X_enc, y, models, n_folds=N_FOLDS):
    cv         = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    model_names= list(models.keys())
    oof_matrix = np.zeros((len(y), len(models)))
    fold_aucs  = {n: [] for n in model_names}

    print(f"\n  {chr(8212)*52}")
    header = f"  {'模型':<18}" + "".join(f"Fold{i+1:>4}" for i in range(n_folds)) + f"  均值"
    print(header)
    print(f"  {chr(8212)*52}")

    for fold, (tr_i, va_i) in enumerate(cv.split(X_enc, y), 1):
        X_tr, X_va = X_enc.iloc[tr_i], X_enc.iloc[va_i]
        y_tr, y_va = y.iloc[tr_i], y.iloc[va_i]
        sw = compute_sample_weight("balanced", y_tr)
        for i, (name, tmpl) in enumerate(models.items()):
            m = type(tmpl)(**tmpl.get_params())
            if "XGB" in name:
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                try: m.fit(X_tr, y_tr, sample_weight=sw)
                except: m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_va)[:, 1]
            oof_matrix[va_i, i] = proba
            fold_aucs[name].append(roc_auc_score(y_va, proba))

    for name in model_names:
        aucs = fold_aucs[name]
        print(f"  {name:<18}" + "".join(f"{a:>7.4f}" for a in aucs) + f"  {np.mean(aucs):.4f}")
    print(f"  {chr(8212)*52}")

    meta = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    meta.fit(oof_matrix, y)
    blend = meta.predict_proba(oof_matrix)[:, 1]
    blend_pred = (blend >= 0.5).astype(int)

    metrics = {}
    for i, name in enumerate(model_names):
        metrics[name] = {
            "auc": round(roc_auc_score(y, oof_matrix[:, i]), 4),
            "f1":  round(f1_score(y, (oof_matrix[:, i] >= 0.5).astype(int)), 4),
        }
    metrics["Stacking"] = {
        "auc":    round(roc_auc_score(y, blend), 4),
        "f1":     round(f1_score(y, blend_pred), 4),
        "pr_auc": round(average_precision_score(y, blend), 4),
    }
    n_top10 = max(1, int(len(y) * 0.10))
    top_idx = np.argsort(-blend)[:n_top10]
    metrics["business"] = {
        "precision_top10pct": round(float(y.iloc[top_idx].mean()), 4),
        "recall_top10pct":    round(float(y.iloc[top_idx].sum() / y.sum()), 4),
    }
    return {"oof_matrix": oof_matrix, "blend_proba": blend,
            "meta": meta, "model_names": model_names, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def train(train_csv: str, output_dir: str = "."):
    t0 = time.time()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    print("="*58)
    print("  车险欺诈检测 · 模型训练  (作者：杨磊磊)")
    print("="*58)

    print(f"\n[1/6] 加载数据: {train_csv}")
    df = pd.read_csv(train_csv)
    y  = df["fraud"]
    print(f"  样本={len(y)}  欺诈率={y.mean():.2%}")

    print("\n[2/6] 特征工程…")
    X = engineer(df.drop(columns=["fraud"]))
    for c in DROP_COLS:
        if c in X.columns: X = X.drop(columns=[c])
    print(f"  特征数: {X.shape[1]}")

    print("\n[3/6] 编码…")
    ALL_OHE  = ["severity_type_combo"] + [c for c in LOW_CARD_COLS if c in X.columns]
    ALL_TE   = [c for c in HIGH_CARD_COLS if c in X.columns]
    num_cols = [c for c in X.select_dtypes(include=np.number).columns
                if c not in ALL_OHE and c not in ALL_TE]
    te  = TargetEncoder(cols=ALL_TE, smoothing=10.0)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
    X_te    = te.fit(X[ALL_TE], y).transform(X[ALL_TE])
    ohe.fit(X[ALL_OHE].astype(str))
    ohe_arr = ohe.transform(X[ALL_OHE].astype(str))
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(ALL_OHE), index=X.index)
    X_enc   = pd.concat([X[num_cols], X_te, ohe_df], axis=1)
    print(f"  编码后: {X_enc.shape[1]} 维")

    print(f"\n[4/6] OOF Stacking ({N_FOLDS} 折)…")
    spw = float((y==0).sum() / (y==1).sum())
    models = {}
    if XGB_AVAILABLE: models["XGBoost"] = build_xgb(scale_pos_weight=spw)
    models["HGB"]          = build_hgb()
    models["RandomForest"] = build_rf()
    sr = oof_stacking(X_enc, y, models)

    print(f"\n  模型对比:")
    for name, m in sr["metrics"].items():
        if name == "business": continue
        tag = " ◀ 最终" if name == "Stacking" else ""
        print(f"  {name:<18} AUC={m.get('auc','—'):>6}  F1={m.get('f1','—'):>6}{tag}")
    bm = sr["metrics"]["business"]
    print(f"  查准率@Top10%={bm['precision_top10pct']:.2%}  召回率@Top10%={bm['recall_top10pct']:.2%}")

    print("\n[5/6] 全集重训练…")
    sw_full = compute_sample_weight("balanced", y)
    final = {}
    for name, tmpl in models.items():
        m = type(tmpl)(**tmpl.get_params())
        if "XGB" in name: m.fit(X_enc, y)
        else:
            try: m.fit(X_enc, y, sample_weight=sw_full)
            except: m.fit(X_enc, y)
        final[name] = m
        print(f"  {name} ✓")

    rf_m = final.get("RandomForest")
    feat_imp = pd.Series(rf_m.feature_importances_, index=X_enc.columns).sort_values(ascending=False) if rf_m else pd.Series(dtype=float)

    print("\n[6/6] 保存…")
    artifacts = {
        "te": te, "ohe": ohe,
        "hgb": final.get("HGB"), "rf": final.get("RandomForest"), "xgb": final.get("XGBoost"),
        "meta": sr["meta"],
        "feature_names": list(X_enc.columns),
        "num_cols": num_cols, "ohe_cols": ALL_OHE, "te_cols": ALL_TE,
        "model_names": list(final.keys()),
        "metrics": {
            "oof_auc":   sr["metrics"].get("HGB", {}).get("auc", 0),
            "blend_auc": sr["metrics"]["Stacking"]["auc"],
            "blend_f1":  sr["metrics"]["Stacking"]["f1"],
            **{f"{k}_auc": v["auc"] for k, v in sr["metrics"].items() if k not in ("Stacking","business")},
        },
        "fraud_rate": float(y.mean()), "train_size": int(len(y)),
        "feat_imp": feat_imp.head(20).round(4).to_dict() if len(feat_imp) else {},
    }
    pkl = out/"model_artifacts.pkl"
    with open(pkl,"wb") as f: pickle.dump(artifacts, f)

    meta_info = {
        "metrics": artifacts["metrics"], "fraud_rate": artifacts["fraud_rate"],
        "train_size": artifacts["train_size"],
        "top_features": list(feat_imp.head(10).index) if len(feat_imp) else [],
        "feat_imp_values": feat_imp.head(10).round(4).to_dict() if len(feat_imp) else {},
        "model_names": list(final.keys()),
        "stacking_detail": {k: v for k, v in sr["metrics"].items()},
    }
    jpath = out/"meta_info.json"
    with open(jpath,"w") as f: json.dump(meta_info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*58}")
    print(f"  ✅ 完成！耗时 {time.time()-t0:.1f}s")
    print(f"  Stacking AUC={sr['metrics']['Stacking']['auc']}  F1={sr['metrics']['Stacking']['f1']}")
    if XGB_AVAILABLE:
        print(f"  XGBoost AUC={sr['metrics']['XGBoost']['auc']}")
    print(f"  查准率@Top10%={bm['precision_top10pct']:.2%}")
    print(f"{'='*58}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--out",   default=".")
    a = p.parse_args()
    train(a.train, a.out)
