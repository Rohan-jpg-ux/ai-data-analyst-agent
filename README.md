# 📊 AI Data Analyst Agent

> Ask any question about your data in plain English — get instant analysis, charts, and explanations.

## 🎯 What It Does

Upload any dataset, then just ask questions like:

- "What's the average sales by region?"
- "Show me the trend in revenue over time"
- "Which category has the highest profit margin?"
- "Is there a correlation between age and income?"

The agent figures out the right pandas analysis, runs it safely, self-corrects if it errors, generates a chart when helpful, and explains the result in plain business language.

## 🏗️ Architecture

Natural Language Question -> Load Schema -> Plan Analysis (Llama 3) -> Generate pandas Code -> Execute Safely -> Retry on error (up to 2x) -> Explain Answer

## 🚀 Quick Start

pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key_here
streamlit run app.py

## 💡 Example Questions

- What is the average revenue by region?
- Show me the distribution of age as a chart
- Which product category sells the most?
- Is there a correlation between price and rating?

## 🧪 Tests

pytest tests/ -v

## ☁️ Deploy

Streamlit Cloud -> app.py -> add GROQ_API_KEY secret

---

Built with LangGraph + Llama 3.3 + Pandas + Streamlit
