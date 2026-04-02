# 🔍 车险欺诈检测系统

基于机器学习（HGB + RandomForest Blending）的车险理赔欺诈识别 Streamlit 应用。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 训练模型（生成 model_artifacts.pkl）

```bash
python train_model.py --train train.csv --out .
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

---

## 📁 项目结构

```
fraud-detection/
├── app.py                # Streamlit 主应用
├── train_model.py        # 模型训练脚本
├── requirements.txt      # 依赖列表
├── meta_info.json        # 模型摘要（提交到 Git）
├── model_artifacts.pkl   # 训练产物（不提交，本地生成）
├── .gitignore
└── README.md
```

---

## 🤖 模型说明

| 项目 | 详情 |
|---|---|
| 基础学习器 | HistGradientBoosting + RandomForest |
| 集成方式 | OOF Blending + Logistic Meta |
| OOF AUC | 0.8587 |
| Blend AUC | 0.8535 |
| Blend F1 | 0.7322 |
| 查准率@Top10% | 62.9% |
| 特征数量 | 84 原始+工程 → 138 编码后 |

### 编码策略

- **Target Encoding**（平滑系数=10）：`insured_occupation`, `insured_hobbies`, `auto_make`, `auto_model`
- **One-Hot Encoding**：`incident_severity`, `incident_type`, `policy_state` 等低基数列
- **丢弃**：`insured_zip`（超高基数，TE 无效）

---

## 🌐 部署到 Streamlit Cloud

1. 将项目 push 到 GitHub
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 仓库，选择 `app.py`
4. 在 **Secrets** 中无需额外配置
5. 点击 **Deploy** 即可

> ⚠️ `model_artifacts.pkl` 不提交 Git，需在 Streamlit Cloud 的 **Files** 面板上传，
> 或在 `packages.txt` 中添加训练步骤。

---

## 🔑 关键欺诈信号

| 信号 | 欺诈率 | 说明 |
|---|---|---|
| 事故严重 = Major Damage | 60.0% | 最强单特征信号 |
| 爱好 = chess | 88.2% | 极端异常 |
| 爱好 = cross-fit | 66.7% | 显著高风险 |
| 职业 = exec-managerial | 38.6% | 高管群体风险高 |
| 无目击者 + 凌晨出险 | — | 行为双重隐蔽信号 |

---

## 📄 License

MIT
