"""
AI Data Analyst Agent — Streamlit UI
LangGraph + Llama 3 — ask questions about your data in plain English
"""

import os
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main, .stApp { background-color: #0f1117; }
.hero { font-size:2.5rem; font-weight:800;
  background:linear-gradient(135deg,#6c63ff,#43b89c);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#888; font-size:1rem; margin-bottom:1.5rem; }
.card { background:#1e2130; border:1px solid #2d3148; border-radius:12px; padding:20px 24px; margin:10px 0; }
.qa-card { background:#1a1d2e; border:1px solid #2d3148; border-radius:12px; padding:18px 22px; margin:12px 0; }
.q-label { color:#6c63ff; font-weight:700; font-size:.9rem; margin-bottom:4px; }
.a-label { color:#43b89c; font-weight:700; font-size:.9rem; margin:10px 0 4px 0; }
.answer-text { color:#e0e0e0; font-size:1.02rem; line-height:1.5; }
div[data-testid="stSidebar"] { background:#151825; }
.stButton>button { background:linear-gradient(135deg,#6c63ff,#43b89c);
  color:#fff; border:none; border-radius:8px; padding:12px 28px; font-weight:700; width:100%; }
.stTextInput input { background:#1e2130 !important; color:#e0e0e0 !important; border:1px solid #2d3148 !important; }
.example-btn { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Data Analyst Agent")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("---")
    st.markdown("### 🔗 Pipeline")
    for step in ["📥 Load Schema", "🧠 Plan Analysis", "✍️ Generate Code", "▶️ Execute", "🔄 Retry if needed", "📝 Explain Answer"]:
        st.markdown(step)

    st.markdown("---")
    st.markdown("**Stack:** LangGraph · Llama 3 · Pandas")

    if st.button("🗑️ Clear conversation"):
        st.session_state["qa_history"] = []
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">📊 AI Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Ask any question about your data in plain English · LangGraph + Llama 3</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json", "parquet"])
with col2:
    st.markdown("#### Or use a sample")
    sample = st.selectbox("Sample datasets", ["None", "Tips", "Titanic", "Iris", "Diamonds"], label_visibility="collapsed")

dataset_path = None
df_preview = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as f:
        f.write(uploaded_file.getbuffer())
        dataset_path = f.name
    if uploaded_file.name.endswith(".csv"):
        df_preview = pd.read_csv(dataset_path)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df_preview = pd.read_excel(dataset_path)
    elif uploaded_file.name.endswith(".json"):
        df_preview = pd.read_json(dataset_path)
elif sample != "None":
    import seaborn as sns
    try:
        df_preview = sns.load_dataset(sample.lower())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
            df_preview.to_csv(f.name, index=False)
            dataset_path = f.name
    except Exception as e:
        st.error(f"Could not load sample: {e}")

if df_preview is not None:
    with st.expander(f"👁️ Preview — {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns", expanded=False):
        st.dataframe(df_preview.head(8), use_container_width=True)
        st.caption(f"Columns: {', '.join(df_preview.columns.tolist())}")

# ── Init chat history ──────────────────────────────────────────────────────────
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# ── Examples (dynamic based on columns) ───────────────────────────────────────
if dataset_path:
    st.markdown("---")
    cols = df_preview.columns.tolist()
    numeric_cols = df_preview.select_dtypes(include="number").columns.tolist()
    cat_cols = df_preview.select_dtypes(include=["object", "category"]).columns.tolist()

    examples = []
    if numeric_cols:
        examples.append(f"What is the average {numeric_cols[0]}?")
        examples.append(f"Show me the distribution of {numeric_cols[0]} as a chart")
    if cat_cols and numeric_cols:
        examples.append(f"What is the total {numeric_cols[0]} by {cat_cols[0]}?")
        examples.append(f"Which {cat_cols[0]} has the highest {numeric_cols[0]}?")
    if len(numeric_cols) >= 2:
        examples.append(f"Is there a correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
    examples.append("What are the key trends in this data?")

    st.markdown("**💡 Try asking:**")
    cols_ui = st.columns(3)
    for i, ex in enumerate(examples[:6]):
        with cols_ui[i % 3]:
            if st.button(ex, key=f"ex{i}"):
                st.session_state["pending_question"] = ex

    # ── Question Input ────────────────────────────────────────────────────────
    st.markdown("---")
    question = st.text_input(
        "💬 Ask a question about your data",
        value=st.session_state.get("pending_question", ""),
        placeholder="e.g. What's the average value by category? Show me a trend over time...",
        key="question_input",
    )
    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]

    ask = st.button("🚀 Ask", use_container_width=True)

    if ask and question.strip():
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ Add your Groq API key in the sidebar.")
            st.stop()

        with st.spinner("🤖 Analyzing..."):
            try:
                from src.agents.analyst_agent import run_analyst
                state = run_analyst(question, dataset_path)

                qa_entry = {
                    "question": question,
                    "answer": state.get("answer", "No answer generated"),
                    "chart_path": state.get("chart_path"),
                    "code": state.get("generated_code", ""),
                    "result": state.get("execution_result", {}).get("result"),
                    "success": state.get("execution_result", {}).get("success", False),
                    "retries": state.get("retry_count", 0),
                }
                st.session_state["qa_history"].insert(0, qa_entry)
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

    # ── Conversation History ──────────────────────────────────────────────────
    if st.session_state["qa_history"]:
        st.markdown("---")
        st.markdown("### 💬 Analysis History")

        for i, qa in enumerate(st.session_state["qa_history"]):
            st.markdown(f"""<div class="qa-card">
            <div class="q-label">❓ {qa['question']}</div>
            <div class="a-label">{'✅ Answer' if qa['success'] else '⚠️ Issue'}</div>
            <div class="answer-text">{qa['answer']}</div>
            </div>""", unsafe_allow_html=True)

            if qa.get("chart_path") and os.path.exists(qa["chart_path"]):
                st.image(qa["chart_path"], use_column_width=True)

            with st.expander(f"🔍 View analysis code & raw result"):
                if qa.get("code"):
                    st.code(qa["code"], language="python")
                if qa.get("result") is not None:
                    st.markdown("**Raw result:**")
                    st.write(qa["result"])
                if qa.get("retries", 0) > 0:
                    st.caption(f"⚠️ Took {qa['retries']} retry attempt(s) to succeed")

else:
    st.markdown("---")
    st.markdown("### 🎯 What you can ask")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card"><b>📈 Aggregations</b><br>
        <span style="color:#888">"What's the average revenue by region?"<br>"Total sales last quarter?"</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><b>📊 Trends & Charts</b><br>
        <span style="color:#888">"Show me sales over time"<br>"Plot the distribution of age"</span></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><b>🔍 Comparisons</b><br>
        <span style="color:#888">"Which category performs best?"<br>"Is there a correlation between X and Y?"</span></div>""", unsafe_allow_html=True)

    st.info("👆 Upload a dataset or pick a sample above to get started")
