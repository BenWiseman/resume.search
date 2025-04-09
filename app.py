import streamlit as st
import os
import tempfile
import io
import zipfile
import numpy as np
import pandas as pd

# Import your existing functions/classes.
from vector_db import VectorDB, summarize_resume  # assumes embed and search methods are defined there
from embed import embed_text_openai
from docx import Document
from pdfminer.high_level import extract_text
import json

# ----- Helper: Parse a resume file based on extension -----
def parse_file(file_obj, filename):
    """
    Parse an uploaded file (docx, pdf, txt, json) and return its text.
       +-------------+
       | file_obj    |
       | filename    |
       +------+------+
              │
         Determine ext
              │
         Return text
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".docx":
        try:
            doc = Document(file_obj)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            st.error(f"Error parsing DOCX file {filename}: {e}")
            return None
    elif ext == ".pdf":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            text = extract_text(tmp_path)
            os.remove(tmp_path)
            return text
        except Exception as e:
            st.error(f"Error parsing PDF file {filename}: {e}")
            return None
    elif ext == ".txt":
        try:
            file_obj.seek(0)
            return file_obj.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Error reading TXT file {filename}: {e}")
            return None
    elif ext == ".json":
        try:
            file_obj.seek(0)
            data = json.load(file_obj)
            sections = []
            # Process basics
            if "basics" in data:
                basics = data["basics"]
                basics_text = []
                for field in ["name", "label", "summary"]:
                    if field in basics and basics[field]:
                        basics_text.append(str(basics[field]))
                if basics_text:
                    sections.append("\n".join(basics_text))
            # Process additional sections
            possible_sections = [
                "work", "volunteer", "education", "awards",
                "publications", "skills", "languages",
                "interests", "references"
            ]
            for s in possible_sections:
                if s in data and isinstance(data[s], list):
                    for entry in data[s]:
                        entry_text = []
                        for _, val in entry.items():
                            if isinstance(val, str) and val.strip():
                                entry_text.append(val.strip())
                        if entry_text:
                            sections.append("\n".join(entry_text))
            return "\n".join(sections)
        except Exception as e:
            st.error(f"Error parsing JSON file {filename}: {e}")
            return None
    else:
        st.warning(f"Unsupported file type: {filename}")
        return None

# ----- Main App -----
def main():
    st.image("https://guddge.com/wp-content/uploads/2023/11/logo_final_blue-1.png?w=250&h=70")
    st.title("Resume Search")
    st.markdown("Automatically search resumes against a job description. [Provided by Guddge LLC](https://guddge.com)")

    # Create tabs for navigation
    setup_tab, query_tab, results_tab = st.tabs(["Setup", "Query", "Results"])

    # ----- Setup Tab -----
    with setup_tab:
        #st.header("## Setup: Upload and Embed Resumes")

        st.markdown("### 1. Insert OpenAI API key")

        st.markdown("(You can find your API key in your [OpenAI account settings](https://platform.openai.com/api-keys).)")
        default_api_key = os.environ.get("OPENAI_API_KEY", "")
        api_key_input = st.text_input("OpenAI API Key:", value=default_api_key if default_api_key else "", disabled=False)
        #saving api key to session state
        if api_key_input:
            st.session_state.api_key = api_key_input

        st.divider()

        st.markdown("### 2. Upload resumes")

        st.markdown("supported types: **.docx**, **.pdf**, **.txt**, **.json**) then press 'Parse & Embed Resumes'.")
        uploaded_files = st.file_uploader("Select resume files", type=["docx", "pdf", "txt", "json"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write("**Uploaded files:**")
            for f in uploaded_files:
                st.write(f.name)
                
        if st.button("Parse & Embed Resumes", key="setup_btn"):
            if not uploaded_files:
                st.error("Please upload resume files first!")
            else:
                resume_docs = []  # list of dicts with keys "path" and "text"
                texts = []       # list of resume texts for embedding
                for file in uploaded_files:
                    file.seek(0)
                    text = parse_file(file, file.name)
                    if text:
                        resume_docs.append({"file": file.name, "text": text})
                        texts.append(text)
                if not texts:
                    st.error("No valid resume texts could be extracted.")
                else:
                    with st.spinner("Embedding resumes..."):
                        try:
                            embeddings_list = embed_text_openai(texts, model="text-embedding-3-large", api_key=st.session_state.api_key)
                            embeddings_array = np.array(embeddings_list, dtype=np.float32)
                            if embeddings_list:
                                d = len(embeddings_list[0])
                            else:
                                st.error("No embeddings returned.")
                                return
                            vector_db = VectorDB(dimension=d)
                            vector_db.add_documents(embeddings_array, resume_docs)
                            st.session_state.vector_db = vector_db
                            st.session_state.resume_docs = resume_docs
                            st.success("Resumes parsed successfully!")
                        except Exception as e:
                            st.error(f"Embedding failed: {e}")

    # ----- Query Tab -----
    with query_tab:
        st.markdown("### 3. Search Resumes")
        if "vector_db" not in st.session_state:
            st.warning("Please complete the Setup first by parsing and embedding resumes.")
        else:
            job_description = st.text_area("Job Description / Query", height=150, placeholder="Enter job description or query here...")
            top_k = st.number_input("Number of resumes to retrieve", value=3, min_value=1, step=1)
            max_tokens = 500#st.slider("Summary Size (max tokens)", min_value=50, max_value=1000, value=500, step=50)
            
            if st.button("Search", key="query_btn"):
                if not job_description:
                    st.error("Please enter a job description or query.")
                else:
                    with st.spinner("Searching resumes..."):
                        try:
                            vector_db = st.session_state.vector_db
                            query_embed_list = embed_text_openai([job_description], model="text-embedding-3-large", api_key=st.session_state.api_key)
                            if not query_embed_list:
                                st.error("No embeddings returned for the query.")
                                return
                            # Ensure the query embedding is a NumPy array   
                            query_embed = np.array(query_embed_list, dtype=np.float32)
                            # Search the vector DB
                            results = vector_db.search(query_embed, top_k=int(top_k))
                            if not results:
                                st.warning("No matching resumes found.")
                                return
                            # Summarize results
                            summarized_results = []
                            for res in results:
                                try:
                                    summary = summarize_resume(
                                        resume_text=res["text"],
                                        query=job_description,
                                        model="gpt-4o",
                                        api_key=st.session_state.api_key,
                                        tokens=max_tokens
                                    )
                                except Exception as e:
                                    summary = f"Summary failed: {e}"
                                res["summary"] = summary
                                res["distance"] = round(res["distance"], 2)
                                summarized_results.append(res)
                            st.session_state.search_results = summarized_results
                            st.success("Search complete!")
                        except Exception as e:
                            st.error(f"Search failed: {e}")

    # ----- Results Tab -----
    with results_tab:
        st.markdown("### Results")
        if "search_results" not in st.session_state or not st.session_state.search_results:
            st.info("No search results yet. Please run a query in the Query tab.")
        else:
            results = st.session_state.search_results
            df = pd.DataFrame(results)
            st.subheader("Top Matching Resumes")
            st.markdown("Double click on the summary text cell to view the resume summary.")
            st.dataframe(df[["file", "summary"]])
            
            #if st.button("Download Top Resumes as Zip", key="download_btn"):
            #    zip_buffer = io.BytesIO()
            #    with zipfile.ZipFile(zip_buffer, "w") as zf:
            #        for res in results:
            #            zf.writestr(res["file"], res["text"])
            #    zip_buffer.seek(0)
                # This produces broken files, removing as it is not actually needed anyway
                #st.download_button("Download Zip", data=zip_buffer, file_name=f"top_resumes_{pd.Timestamp.now().date()}.zip")

if __name__ == "__main__":
    main()