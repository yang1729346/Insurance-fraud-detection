"""
车险欺诈检测系统 · Streamlit App
作者：杨磊磊
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date, timedelta


from sklearn.base import BaseEstimator, TransformerMixin

# ══════════════════════════════════════════════════════════════════════════════
# TargetEncoder — 必须在 app.py 中定义，才能正确反序列化 model_artifacts.pkl
# （pickle 要求序列化时的类定义与加载时在同一命名空间）
# ══════════════════════════════════════════════════════════════════════════════

class TargetEncoder(BaseEstimator, TransformerMixin):
    """贝叶斯平滑 Target Encoding。公式：encoded=(n×mean + m×global_mean)/(n+m)"""
    def __init__(self, cols: list, smoothing: float = 10.0):
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
                out[col].astype(str).map(self._stats[col])
                .fillna(self._global_mean).astype(float)
            )
        return out

# ── 页面配置（必须第一行）──────────────────────────────────────────────────────
st.set_page_config(
    page_title="车险欺诈检测系统 · 杨磊磊",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = Path(__file__).parent / "model_artifacts.pkl"
META_PATH     = Path(__file__).parent / "meta_info.json"

HIGH_RISK_OCC = {"handlers-cleaners", "machine-op-inspct", "transport-moving",
                 "craft-repair", "farming-fishing"}
HIGH_RISK_HOB = {"skydiving", "bungie-jumping", "base-jumping",
                 "chess", "cross-fit", "paintball", "board-games"}

# ── 下拉选项 ──────────────────────────────────────────────────────────────────
# ── 原始英文值（提交给模型时使用）────────────────────────────────────────────
OPTIONS = {
    "policy_state":           ["A", "B", "C"],
    "policy_csl":             ["100/300", "250/500", "500/1000"],
    "insured_sex":            ["FEMALE", "MALE"],
    "insured_education_level":["Associate", "College", "High School", "JD", "MD", "Masters", "PhD"],
    "insured_occupation":     ["adm-clerical", "armed-forces", "craft-repair", "exec-managerial",
                               "farming-fishing", "handlers-cleaners", "machine-op-inspct",
                               "other-service", "priv-house-serv", "prof-specialty",
                               "protective-serv", "sales", "tech-support", "transport-moving"],
    "insured_hobbies":        ["base-jumping", "basketball", "board-games", "bungie-jumping",
                               "camping", "chess", "cross-fit", "dancing", "exercise", "golf",
                               "hiking", "kayaking", "movies", "paintball", "polo",
                               "reading", "skydiving", "sleeping", "video-games", "yachting"],
    "insured_relationship":   ["husband", "not-in-family", "other-relative",
                               "own-child", "unmarried", "wife"],
    "incident_type":          ["Multi-vehicle Collision", "Parked Car",
                               "Single Vehicle Collision", "Vehicle Theft"],
    "collision_type":         ["Front Collision", "Rear Collision", "Side Collision", "Unknown"],
    "incident_severity":      ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"],
    "authorities_contacted":  ["Ambulance", "Fire", "Other", "Police"],
    "incident_state":         ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    "incident_city":          ["Arlington", "Columbus", "Hillsdale", "Northbend",
                               "Northbrook", "Riverwood", "Springfield"],
    "property_damage":        ["NO", "YES", "Unknown"],
    "police_report_available":["NO", "YES", "Unknown"],
    "auto_make":              ["Accura", "Audi", "BMW", "Chevrolet", "Dodge", "Ford",
                               "Honda", "Jeep", "Mercedes", "Nissan", "Saab",
                               "Suburu", "Toyota", "Volkswagen"],
    "auto_model":             ["3 Series", "92x", "93", "95", "A3", "A5", "Accord",
                               "C300", "CRV", "Camry", "Civic", "Corolla", "E400",
                               "Escape", "F150", "Forrestor", "Fusion", "Grand Cherokee",
                               "Highlander", "Impreza", "Jetta", "Legacy", "M5", "MDX",
                               "ML350", "Malibu", "Maxima", "Neon", "Passat", "Pathfinder",
                               "RAM", "RSX", "Silverado", "TL", "Tahoe", "Ultima",
                               "Wrangler", "X5", "X6"],
}

# ── 中文显示标签（UI 展示用，selectbox format_func 映射回英文原始值）──────────
CN_LABELS = {
    # 性别
    "FEMALE": "女", "MALE": "男",
    # 学历
    "Associate": "大专", "College": "本科", "High School": "高中",
    "JD": "法学博士", "MD": "医学博士", "Masters": "硕士", "PhD": "博士",
    # 职业
    "adm-clerical": "行政文员", "armed-forces": "军人", "craft-repair": "工匠维修",
    "exec-managerial": "高管", "farming-fishing": "农渔业", "handlers-cleaners": "搬运清洁",
    "machine-op-inspct": "机械操作", "other-service": "其他服务", "priv-house-serv": "家政服务",
    "prof-specialty": "专业技术", "protective-serv": "安保服务", "sales": "销售",
    "tech-support": "技术支持", "transport-moving": "运输物流",
    # 爱好
    "base-jumping": "定点跳伞", "basketball": "篮球", "board-games": "桌游",
    "bungie-jumping": "蹦极", "camping": "露营", "chess": "国际象棋",
    "cross-fit": "健身训练", "dancing": "舞蹈", "exercise": "健身", "golf": "高尔夫",
    "hiking": "徒步", "kayaking": "皮划艇", "movies": "看电影", "paintball": "彩弹射击",
    "polo": "马球", "reading": "阅读", "skydiving": "跳伞", "sleeping": "睡觉",
    "video-games": "电子游戏", "yachting": "帆船",
    # 与被保险人关系
    "husband": "配偶（夫）", "not-in-family": "非家庭成员", "other-relative": "其他亲属",
    "own-child": "子女", "unmarried": "未婚伴侣", "wife": "配偶（妻）",
    # 事故类型
    "Multi-vehicle Collision": "多车碰撞", "Parked Car": "停放车辆事故",
    "Single Vehicle Collision": "单车碰撞", "Vehicle Theft": "车辆盗窃",
    # 碰撞类型
    "Front Collision": "正面碰撞", "Rear Collision": "追尾碰撞",
    "Side Collision": "侧面碰撞", "Unknown": "未知",
    # 事故严重程度
    "Major Damage": "严重损毁", "Minor Damage": "轻微损毁",
    "Total Loss": "全损", "Trivial Damage": "轻微擦伤",
    # 联系当局
    "Ambulance": "救护车", "Fire": "消防", "Other": "其他", "Police": "警察",
    # 财产损失 / 警察报告
    "NO": "否", "YES": "是",
    # 保单州（字母代号，保留）
    "A": "A州", "B": "B州", "C": "C州",
}

def cn(val):
    """将原始英文选项值翻译为中文显示标签，无匹配则原样返回。"""
    return CN_LABELS.get(val, val)

def cn_selectbox(label, widget_key, options):
    """显示中文、返回原始英文值的 selectbox 封装。
    widget_key: Streamlit widget key，保证每个控件唯一。
    """
    labels = [cn(o) for o in options]
    idx = st.selectbox(label, range(len(options)),
                       format_func=lambda i: labels[i],
                       key=widget_key)
    return options[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 缓存资源加载
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="加载模型中…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 特征工程（与训练代码保持一致）
# ══════════════════════════════════════════════════════════════════════════════

def engineer_single(row: dict) -> pd.DataFrame:
    """将单条表单输入转为模型特征 DataFrame（1行）。"""
    df = pd.DataFrame([row])

    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    df["incident_date"]    = pd.to_datetime(df["incident_date"],    errors="coerce")
    df["policy_end_date"]  = df["policy_bind_date"] + pd.DateOffset(years=1)

    # 缺失标记
    df["collision_type"] = df["collision_type"].replace("?", "Unknown")

    # 时间特征
    df["days_since_policy_start"] = (df["incident_date"] - df["policy_bind_date"]).dt.days.astype(float)
    df["days_to_policy_end"]      = (df["policy_end_date"] - df["incident_date"]).dt.days.astype(float)
    df["is_night_incident"]       = df["incident_hour_of_the_day"].isin([0,1,2,3,4,5,22,23]).astype(int)
    df["is_weekend_incident"]     = (df["incident_date"].dt.dayofweek >= 5).astype(int)
    df["is_early_claim"]          = (df["days_since_policy_start"] < 30).astype(int)
    df["is_near_expiry_claim"]    = (df["days_to_policy_end"] < 30).astype(int)

    # 金额特征
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

    # 行为特征
    df["no_witness"]              = (df["witnesses"] == 0).astype(int)
    df["no_police_report"]        = df["police_report_available"].isin(["NO", "Unknown"]).astype(int)
    df["no_property_damage_rec"]  = df["property_damage"].isin(["NO", "Unknown"]).astype(int)
    df["evidence_gap_score"]      = df["no_witness"] + df["no_police_report"] + df["no_property_damage_rec"]

    # 风险画像
    df["is_high_risk_occ"]        = df["insured_occupation"].isin(HIGH_RISK_OCC).astype(int)
    df["is_high_risk_hobby"]      = df["insured_hobbies"].isin(HIGH_RISK_HOB).astype(int)
    df["high_risk_profile"]       = (df["is_high_risk_occ"] & df["is_high_risk_hobby"]).astype(int)
    df["is_extreme_hobby"]        = df["insured_hobbies"].isin(
        {"skydiving", "bungie-jumping", "base-jumping", "chess", "cross-fit"}
    ).astype(int)

    # 组合特征
    df["severity_type_combo"]     = (
        df["incident_severity"].str.replace(" ", "_") + "__" +
        df["incident_type"].str.replace(" ", "_")
    )
    df["net_capital"]             = df["capital-gains"] + df["capital-loss"]
    df["night_no_witness"]        = (df["is_night_incident"] & df["no_witness"]).astype(int)
    df["collision_type_missing"]  = (df["collision_type"] == "Unknown").astype(int)

    return df


def encode_and_predict(df_feat: pd.DataFrame, artifacts: dict) -> tuple[float, dict]:
    """编码 + 模型预测，返回 (欺诈概率, 风险因子字典)。"""
    te  = artifacts["te"]
    ohe = artifacts["ohe"]
    hgb = artifacts["hgb"]
    rf  = artifacts["rf"]
    meta = artifacts["meta"]
    num_cols = artifacts["num_cols"]
    ohe_cols = artifacts["ohe_cols"]
    te_cols  = artifacts["te_cols"]

    # Target Encoding
    te_part  = te.transform(df_feat[[c for c in te_cols if c in df_feat.columns]])
    # One-Hot
    ohe_part = ohe.transform(df_feat[[c for c in ohe_cols if c in df_feat.columns]].astype(str))
    ohe_df   = pd.DataFrame(ohe_part,
                             columns=ohe.get_feature_names_out([c for c in ohe_cols if c in df_feat.columns]),
                             index=df_feat.index)
    # Numeric
    num_part = df_feat[[c for c in num_cols if c in df_feat.columns]].copy().astype(float)

    X_enc = pd.concat([num_part, te_part, ohe_df], axis=1)

    # Align columns
    expected = artifacts["feature_names"]
    for c in expected:
        if c not in X_enc.columns:
            X_enc[c] = 0.0
    X_enc = X_enc[expected]

    # Blend prediction
    hgb_p  = hgb.predict_proba(X_enc)[0, 1]
    rf_p   = rf.predict_proba(X_enc)[0, 1]
    blend  = meta.predict_proba(np.array([[hgb_p, rf_p]]))[0, 1]

    # Risk factors
    sev_raw = df_feat["incident_severity"].iloc[0]
    risk_factors = {
        "事故严重程度":   cn(sev_raw),
        "夜间事故":       bool(df_feat["is_night_incident"].iloc[0]),
        "无目击者":       bool(df_feat["no_witness"].iloc[0]),
        "无警察报告":     bool(df_feat["no_police_report"].iloc[0]),
        "高风险爱好":     bool(df_feat["is_high_risk_hobby"].iloc[0]),
        "高风险职业":     bool(df_feat["is_high_risk_occ"].iloc[0]),
        "理赔/保费比率":  round(float(df_feat["claim_to_premium_ratio"].iloc[0]), 2),
        "证据缺口评分":   int(df_feat["evidence_gap_score"].iloc[0]),
        "保单早期出险":   bool(df_feat["is_early_claim"].iloc[0]),
    }
    return float(blend), risk_factors


# ══════════════════════════════════════════════════════════════════════════════
# UI 组件
# ══════════════════════════════════════════════════════════════════════════════

def risk_gauge(prob: float) -> go.Figure:
    """绘制仪表盘风险图。"""
    color = "#22c55e" if prob < 0.3 else ("#f59e0b" if prob < 0.6 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef9c3"},
                {"range": [60,100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.85, "value": prob * 100},
        },
        title={"text": "欺诈概率", "font": {"size": 14}},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig



# ── 特征名称中文映射（用于图表展示）─────────────────────────────────────────
FEAT_CN = {
    "incident_severity_Major Damage":   "事故严重度：严重损毁",
    "insured_hobbies":                  "投保人爱好",
    "auto_model":                       "车辆型号",
    "severity_type_combo_Major_Damage__Single_Vehicle_Collision": "严重损毁×单车碰撞",
    "incident_severity_Minor Damage":   "事故严重度：轻微损毁",
    "severity_type_combo_Major_Damage__Multi-vehicle_Collision":  "严重损毁×多车碰撞",
    "is_extreme_hobby":                 "极高风险爱好",
    "vehicle_claim":                    "车辆理赔金额",
    "is_high_risk_hobby":               "高风险爱好标记",
    "claim_to_csl_ratio":               "理赔/保额上限比",
    "claim_to_premium_ratio":           "理赔/年保费比",
    "total_claim_amount":               "总理赔金额",
    "incident_severity_Total Loss":     "事故严重度：全损",
    "property_claim":                   "财产理赔金额",
    "claim_over_deductable_x":          "理赔/免赔额倍数",
    "policy_annual_premium":            "年保费",
    "injury_claim":                     "人伤理赔金额",
    "vehicle_to_total_ratio":           "车损占总理赔比",
    "insured_occupation":               "投保人职业",
    "deductable_to_premium":            "免赔额/保费比",
    "vehicle_age":                      "车龄",
    "age":                              "投保人年龄",
    "customer_months":                  "客户月数",
    "net_capital":                      "净资本",
    "days_since_policy_start":          "保单已生效天数",
    "days_to_policy_end":               "距保单到期天数",
    "evidence_gap_score":               "证据缺口评分",
    "no_witness":                       "无目击者",
    "no_police_report":                 "无警察报告",
    "is_night_incident":                "凌晨/夜间出险",
    "is_weekend_incident":              "周末出险",
    "is_early_claim":                   "保单早期出险",
    "high_risk_profile":                "双高风险画像",
    "injury_to_total_ratio":            "人伤占总理赔比",
    "is_high_risk_occ":                 "高风险职业标记",
    "night_no_witness":                 "夜间+无目击者",
}

def feat_cn(name: str) -> str:
    """特征英文名 → 中文名，无匹配则原样返回。"""
    return FEAT_CN.get(name, name)

def feature_importance_chart(feat_imp: dict) -> go.Figure:
    """绘制特征重要性水平条形图。"""
    names = [feat_cn(k) for k in list(feat_imp.keys())[::-1]]
    vals  = list(feat_imp.values())[::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#6366f1",
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont={"size": 10},
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="重要性", yaxis={"tickfont": {"size": 10}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 侧边栏导航
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## 🔍 车险欺诈检测")
        st.markdown("---")
        page = st.radio(
            "导航",
            ["🏠 首页概览", "🔮 单案件预测", "📊 模型分析"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            "<small>模型：HGB + RF Blending<br>"
            "训练集：700条 · 欺诈率 25.9%</small>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<span style='font-size:12px;color:var(--color-text-secondary)'>作者</span><br>"
            "<span style='font-size:15px;font-weight:500'>杨磊磊</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    return page


# ══════════════════════════════════════════════════════════════════════════════
# 页面：首页概览
# ══════════════════════════════════════════════════════════════════════════════

def page_home(meta: dict):
    st.title("🏠 车险欺诈检测系统")
    st.caption("作者：杨磊磊 · 基于机器学习的保险理赔欺诈智能识别平台")
    st.markdown("基于机器学习的保险理赔欺诈智能识别平台，支持实时单案评分与批量风险分析。")

    # KPI 卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("训练集大小",    f"{meta['train_size']} 条")
    c2.metric("训练集欺诈率", f"{meta['fraud_rate']*100:.1f}%")
    c3.metric("模型 AUC",     str(meta['metrics']['blend_auc']))
    c4.metric("F1-Score",     str(meta['metrics']['blend_f1']))

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 系统功能")
        st.markdown("""
| 功能 | 说明 |
|---|---|
| 🔮 **单案件预测** | 实时输入理赔信息，输出欺诈概率与风险因子 |
| 📊 **模型分析** | 查看特征重要性、模型性能指标 |
| 🎯 **风险分级** | 极低 / 低 / 中 / 高 / 极高 五级预警 |
""")
        st.subheader("🚨 欺诈识别维度")
        st.markdown("""
- **时间异常**：凌晨出险、保单早期/临期出险
- **金额异常**：理赔/保费比率虚高、老车高额理赔
- **行为异常**：无目击者、无警察报告、证据链缺失
- **风险画像**：高危职业 × 高危爱好组合
""")

    with col2:
        st.subheader("📈 Top 10 关键特征")
        feat_vals = meta.get("feat_imp_values", {})
        top10 = dict(list(feat_vals.items())[:10])
        st.plotly_chart(feature_importance_chart(top10),
                        use_container_width=True, key="home_fi")


# ══════════════════════════════════════════════════════════════════════════════
# 页面：单案件预测
# ══════════════════════════════════════════════════════════════════════════════

def page_predict(artifacts: dict):
    st.title("🔮 单案件欺诈评分")
    st.markdown("填写理赔案件信息，点击「开始评估」获得实时欺诈概率与风险解析。")

    with st.form("predict_form"):
        # ── 保单信息 ──────────────────────────────────────────────────
        st.markdown("#### 📋 保单信息")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age             = r1c1.number_input("投保人年龄", 18, 70, 38)
        customer_months = r1c2.number_input("客户月数", 0, 600, 200)
        policy_bind_date= r1c3.date_input("保单生效日", value=date(2010, 1, 1),
                                           min_value=date(1990, 1, 1), max_value=date.today())
        policy_state    = cn_selectbox("保单州", "pstate", OPTIONS["policy_state"])

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        policy_csl         = r2c1.selectbox("责任限额(CSL)", OPTIONS["policy_csl"])
        policy_deductable  = r2c2.selectbox("免赔额", [500, 1000, 2000])
        policy_annual_premium = r2c3.number_input("年保费(元)", 400, 2100, 1250)
        umbrella_limit     = r2c4.number_input("保护伞限额", -1000000, 10000000, 0, step=500000)

        # ── 投保人信息 ────────────────────────────────────────────────
        st.markdown("#### 👤 投保人信息")
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        insured_sex        = cn_selectbox("性别", "sex", OPTIONS["insured_sex"])
        insured_education  = cn_selectbox("学历", "edu", OPTIONS["insured_education_level"])
        insured_occupation = cn_selectbox("职业", "occ", OPTIONS["insured_occupation"])
        insured_hobbies    = cn_selectbox("爱好", "hob", OPTIONS["insured_hobbies"])

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        insured_relationship = cn_selectbox("与被保险人关系", "rel", OPTIONS["insured_relationship"])
        insured_zip          = r4c2.number_input("邮编", 400000, 700000, 500000)
        capital_gains        = r4c3.number_input("资本收益", 0, 100000, 0, step=1000)
        capital_loss         = r4c4.number_input("资本损失", -110000, 0, 0, step=1000)

        # ── 事故信息 ──────────────────────────────────────────────────
        st.markdown("#### 🚗 事故信息")
        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        incident_date      = r5c1.date_input("事故日期", value=date(2015, 1, 15),
                                              min_value=date(2010, 1, 1), max_value=date.today())
        incident_hour      = r5c2.slider("事故时刻（小时）", 0, 23, 12)
        incident_type      = cn_selectbox("事故类型", "itype", OPTIONS["incident_type"])
        incident_severity  = cn_selectbox("事故严重程度", "isev", OPTIONS["incident_severity"])

        r6c1, r6c2, r6c3, r6c4 = st.columns(4)
        collision_type     = cn_selectbox("碰撞类型", "ctype", OPTIONS["collision_type"])
        authorities        = cn_selectbox("联系的当局", "auth", OPTIONS["authorities_contacted"])
        incident_state     = r6c3.selectbox("事故发生州", OPTIONS["incident_state"])
        incident_city      = r6c4.selectbox("事故发生城市", OPTIONS["incident_city"])

        r7c1, r7c2, r7c3, r7c4 = st.columns(4)
        n_vehicles         = r7c1.selectbox("涉及车辆数", [1, 2, 3, 4])
        property_damage    = cn_selectbox("财产损失记录", "prop", OPTIONS["property_damage"])
        bodily_injuries    = r7c3.selectbox("人身伤亡数", [0, 1, 2])
        witnesses          = r7c4.selectbox("目击者数", [0, 1, 2, 3])

        r8c1, r8c2 = st.columns(2)
        police_report      = cn_selectbox("警察报告", "police", OPTIONS["police_report_available"])

        # ── 理赔金额 ──────────────────────────────────────────────────
        st.markdown("#### 💰 理赔金额")
        r9c1, r9c2, r9c3, r9c4 = st.columns(4)
        total_claim   = r9c1.number_input("总理赔金额", 0, 200000, 55000, step=1000)
        injury_claim  = r9c2.number_input("人伤理赔", 0, 50000,  6500,  step=500)
        property_claim= r9c3.number_input("财产理赔", 0, 50000,  7000,  step=500)
        vehicle_claim = r9c4.number_input("车辆理赔", 0, 100000, 40000, step=1000)

        # ── 车辆信息 ──────────────────────────────────────────────────
        st.markdown("#### 🚙 车辆信息")
        r10c1, r10c2, r10c3 = st.columns(3)
        auto_make  = r10c1.selectbox("车辆品牌", OPTIONS["auto_make"])
        auto_model = r10c2.selectbox("车辆型号", OPTIONS["auto_model"])
        auto_year  = r10c3.slider("出厂年份", 1995, 2015, 2007)

        submitted = st.form_submit_button("🔍 开始评估", use_container_width=True, type="primary")

    # ── 预测结果 ─────────────────────────────────────────────────────
    if submitted:
        row = {
            "age": age, "customer_months": customer_months,
            "policy_bind_date": str(policy_bind_date), "policy_state": policy_state,
            "policy_csl": policy_csl, "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium, "umbrella_limit": umbrella_limit,
            "insured_zip": insured_zip, "insured_sex": insured_sex,
            "insured_education_level": insured_education, "insured_occupation": insured_occupation,
            "insured_hobbies": insured_hobbies, "insured_relationship": insured_relationship,
            "capital-gains": capital_gains, "capital-loss": capital_loss,
            "incident_date": str(incident_date), "incident_type": incident_type,
            "collision_type": collision_type, "incident_severity": incident_severity,
            "authorities_contacted": authorities, "incident_state": incident_state,
            "incident_city": incident_city, "incident_hour_of_the_day": incident_hour,
            "number_of_vehicles_involved": n_vehicles, "property_damage": property_damage,
            "bodily_injuries": bodily_injuries, "witnesses": witnesses,
            "police_report_available": police_report, "total_claim_amount": total_claim,
            "injury_claim": injury_claim, "property_claim": property_claim,
            "vehicle_claim": vehicle_claim, "auto_make": auto_make,
            "auto_model": auto_model, "auto_year": auto_year,
        }

        with st.spinner("模型评估中…"):
            df_feat = engineer_single(row)
            prob, risk_factors = encode_and_predict(df_feat, artifacts)

        # ── 风险等级 ─────────────────────────────────────────────────
        if prob < 0.2:
            tier, tier_color, tier_emoji = "极低风险", "green",  "✅"
        elif prob < 0.4:
            tier, tier_color, tier_emoji = "低风险",   "green",  "🟢"
        elif prob < 0.6:
            tier, tier_color, tier_emoji = "中等风险", "orange", "🟡"
        elif prob < 0.8:
            tier, tier_color, tier_emoji = "高风险",   "red",    "🔴"
        else:
            tier, tier_color, tier_emoji = "极高风险", "red",    "🚨"

        st.markdown("---")
        st.subheader("📊 评估结果")

        col_g, col_r = st.columns([1, 1.5])
        with col_g:
            st.plotly_chart(risk_gauge(prob), use_container_width=True, key="gauge")
            st.markdown(
                f"<h3 style='text-align:center;color:{tier_color}'>"
                f"{tier_emoji} {tier}</h3>",
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown("#### 🔍 风险因子详情")
            for factor, value in risk_factors.items():
                if isinstance(value, bool):
                    icon  = "⚠️" if value else "✅"
                    color = "red" if value else "green"
                    val_str = "是" if value else "否"
                elif isinstance(value, (int, float)):
                    icon  = "⚠️" if value > 2 else "ℹ️"
                    color = "inherit"
                    val_str = str(value)
                else:
                    icon  = "⚠️" if value in ["严重损毁", "全损"] else "ℹ️"
                    color = "red" if value in ["严重损毁", "全损"] else "inherit"
                    val_str = str(value)

                st.markdown(
                    f"{icon} **{factor}**：<span style='color:{color}'>{val_str}</span>",
                    unsafe_allow_html=True,
                )

        # ── 建议 ─────────────────────────────────────────────────────
        st.markdown("---")
        if prob >= 0.6:
            st.error(
                f"⚠️ **建议人工审核**：该案件欺诈概率 {prob*100:.1f}%，属于{tier}，"
                "建议立即转入人工调查通道，核实事故现场、联系目击者、"
                "复查修理厂评估报告。"
            )
        elif prob >= 0.4:
            st.warning(
                f"🟡 **建议辅助核查**：该案件欺诈概率 {prob*100:.1f}%，属于{tier}，"
                "建议电话回访当事人，核对证据链完整性，必要时安排现场复勘。"
            )
        else:
            st.success(
                f"✅ **建议正常处理**：该案件欺诈概率 {prob*100:.1f}%，属于{tier}，"
                "可进入标准理赔处理流程。"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 页面：模型分析
# ══════════════════════════════════════════════════════════════════════════════

def page_analysis(meta: dict):
    st.title("📊 模型性能分析")

    # 性能指标
    st.subheader("🏆 模型指标（5-Fold OOF Blending）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOF AUC-ROC",  meta["metrics"]["oof_auc"])
    c2.metric("Blend AUC",    meta["metrics"]["blend_auc"])
    c3.metric("Blend F1",     meta["metrics"]["blend_f1"])
    c4.metric("查准率@Top10%", "62.9%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 特征重要性 Top 10")
        st.plotly_chart(
            feature_importance_chart(meta["feat_imp_values"]),
            use_container_width=True, key="analysis_fi"
        )

    with col2:
        st.subheader("🔬 模型架构")
        st.markdown("""
**基础学习器**

| 模型 | OOF AUC | 特点 |
|---|---|---|
| HistGradientBoosting | 0.8587 | 梯度提升，处理混合特征强 |
| RandomForest | 0.8266 | 集成树，方差低，互补 |

**集成策略**：OOF Blending + LogisticRegression 元学习器

**编码策略**

| 类型 | 列 | 方法 |
|---|---|---|
| 超高基数 | 邮编(insured_zip) | 丢弃 |
| 高基数 | 职业、爱好、车型品牌 | 目标编码（平滑=10）|
| 低基数 | 严重程度、事故类型等 | 独热编码 |
""")

        st.subheader("⚡ 特征工程维度")
        dims = {
            "时间特征": 6, "金额特征": 8, "行为特征": 6,
            "风险画像": 5, "组合特征": 4, "缺失标记": 3
        }
        fig = px.bar(
            x=list(dims.values()), y=list(dims.keys()),
            orientation="h", color=list(dims.values()),
            color_continuous_scale="Purples",
            labels={"x": "特征数量", "y": ""},
        )
        fig.update_layout(height=240, showlegend=False,
                          margin=dict(l=10, r=20, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="dims_chart")

    # 业务解读
    st.markdown("---")
    st.subheader("💼 关键欺诈信号解读")
    signals = [
        ("事故严重度 = 严重损毁", "60.0%", "🔴", "严重损失事故的欺诈率是平均水平的 2.3倍"),
        ("爱好 = 国际象棋 / 健身训练", "88% / 67%", "🔴", "特定爱好群体呈现异常欺诈集中"),
        ("职业 = 高管",               "38.6%", "🟡", "高管职业欺诈率显著高于平均"),
        ("目击者 ≥ 2",               "34.1%", "🟡", "多目击者反而欺诈率更高，疑似团伙"),
        ("无警察报告",               "27.1%", "🟡", "回避当局是欺诈的典型行为模式"),
        ("事故类型 = 车辆盗窃",       "7.4%",  "🟢", "车辆盗窃类案件欺诈率最低"),
    ]
    for sig, rate, icon, note in signals:
        with st.container():
            c1, c2, c3 = st.columns([2, 1, 3])
            c1.markdown(f"**{sig}**")
            c2.markdown(f"{icon} `{rate}`")
            c3.markdown(f"<small>{note}</small>", unsafe_allow_html=True)
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    artifacts = load_model()
    meta      = load_meta()
    page      = sidebar()

    if "首页" in page:
        page_home(meta)
    elif "预测" in page:
        page_predict(artifacts)
    elif "分析" in page:
        page_analysis(meta)


if __name__ == "__main__":
    main()
