# app.py

import streamlit as st
import pandas as pd
import joblib
from job_classifier import scrape_karkidi_jobs, classify_new_jobs, notify_user
from datetime import datetime
from sklearn.preprocessing import normalize

# Load model and vectorizer
model = joblib.load("job_cluster_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Step 1: Define human-readable cluster names
cluster_names = {
    0: "🧠 Data Science & ML",
    1: "☁️ Cloud & DevOps",
    2: "🧮 Analytics & Business Intelligence",
    3: "💻 Backend Development",
    4: "📊 General Tech & Others"
}

# 🔹 Step 2: Streamlit UI
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("💼 Job Recommendation System")
st.markdown("Get job updates tailored to your skills and interests using unsupervised ML.")

# Sidebar – User preferences
keyword = st.sidebar.selectbox("🔍 Choose job keyword", ["Data Science", "Cloud", "Machine Learning", "AI", "NLP"])
selected_names = st.sidebar.multiselect("🎯 Select your job interests", list(cluster_names.values()), default=[
    cluster_names[0], cluster_names[2]
])

# Map selected names back to cluster numbers
user_clusters = [k for k, v in cluster_names.items() if v in selected_names]

# 🔍 Scrape and classify jobs
if st.sidebar.button("🚀 Run Job Check"):
    st.subheader(f"🔄 Checking new jobs for: `{keyword}`")
    new_jobs = scrape_karkidi_jobs(keyword=keyword, pages=1)
    classified = classify_new_jobs(new_jobs, model, vectorizer)
    matched = notify_user(classified, user_clusters)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    all_jobs_path = f"results/{keyword}_all_jobs_{now}.csv"
    matched_jobs_path = f"results/{keyword}_matched_jobs_{now}.csv"

    classified.to_csv(all_jobs_path, index=False)
    matched.to_csv(matched_jobs_path, index=False)

    st.subheader("✅ All Classified Jobs")
    st.dataframe(classified)
    st.download_button("⬇️ Download Classified Jobs", data=classified.to_csv(index=False), file_name="classified_jobs.csv")

    st.subheader("🚨 Matched Jobs")
    if not matched.empty:
        st.dataframe(matched)
        st.download_button("⬇️ Download Matched Jobs", data=matched.to_csv(index=False), file_name="matched_jobs.csv")
    else:
        st.info("No matched jobs found for your selected interests.")
else:
    st.info("Use the sidebar to select a keyword and clusters, then click 'Run Job Check'.")
