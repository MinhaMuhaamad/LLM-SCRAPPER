
import streamlit as st
import os
import pandas as pd
st.set_page_config(page_title="NeurIPS Paper Scraper", layout="wide")
st.title("📄 NeurIPS Research Paper Scraper")
def run_scraper():
    # Add your scraping logic here
    pass

if st.button("Start Scraping 🛠️"):
    st.info("⏳ Scraping in progress...")
    run_scraper()
    st.success("✅ Scraping Completed!")
csv_file = "neurips_papers/annotated_papers.csv"
if os.path.exists(csv_file):
    st.subheader("📂 Select a Year to View Papers")
    papers_df = pd.read_csv(csv_file)
    papers_df = papers_df.drop(columns=["Authors"], errors='ignore')
    years = sorted(papers_df["Year"].unique(), reverse=True)
    
    selected_year = st.selectbox("📅 Select Year", ["Select a year"] + list(years), index=0)
    if selected_year != "Select a year":
        year_df = papers_df[papers_df["Year"] == selected_year]
        st.dataframe(year_df, height=300, use_container_width=True)
        selected_paper = st.selectbox(f"📑 Select a paper from {selected_year}", year_df["Title"], key=f"select_{selected_year}")
        if st.button(f"📥 Download {selected_paper}", key=f"download_{selected_year}"):
            paper_path = year_df[year_df["Title"] == selected_paper]["File Path"].values[0]
            with open(paper_path, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name=os.path.basename(paper_path), mime="application/pdf")
    st.subheader("📜 Complete Paper Metadata")
    st.dataframe(papers_df, use_container_width=True)
st.warning("⚠️ Scraping may take time. Be patient while the process completes.")

