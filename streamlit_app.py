import os
import shutil
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import streamlit as st

# Step 1: Function to extract abstracts from PDFs
def extract_abstracts(uploaded_files):
    abstracts = []
    
    for pdf_file in uploaded_files:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        abstract = doc[0].get_text("text")  # Extract text from the first page
        abstracts.append((pdf_file.name, abstract))
        doc.close()
    
    return abstracts

# Step 2: Function to generate embeddings
def generate_embeddings(abstracts):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
    texts = [abstract[1] for abstract in abstracts]
    embeddings = model.encode(texts)
    
    # Create a DataFrame for easy manipulation
    df = pd.DataFrame(embeddings)
    df['filename'] = [abstract[0] for abstract in abstracts]
    return df

# Step 3: Function to cluster embeddings
def cluster_embeddings(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df.iloc[:, :-1])  # Exclude the filename
    return df

# Step 4: Function to organize PDFs
def organize_pdfs(df, base_folder):
    for cluster in df['cluster'].unique():
        cluster_folder = os.path.join(base_folder, f'cluster_{cluster}')
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Move files to the appropriate cluster folder
        for _, row in df[df['cluster'] == cluster].iterrows():
            shutil.move(os.path.join(base_folder, row['filename']), cluster_folder)

# Step 5: Streamlit UI
def main():
    st.title("PDF Organizer Tool")
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    num_clusters = st.number_input("Select Number of Clusters", min_value=1, max_value=20, value=5)

    if st.button("Organize PDFs"):
        if uploaded_files:
            # Create a temporary directory to store uploaded files
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded PDFs to the temporary directory
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            # Process the PDFs
            abstracts = extract_abstracts(uploaded_files)
            df = generate_embeddings(abstracts)
            clustered_df = cluster_embeddings(df, num_clusters)
            organize_pdfs(clustered_df, temp_dir)
            
            st.success("PDFs organized successfully!")
            st.write(f"Check the organized PDFs in the directory: {os.path.abspath(temp_dir)}")
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
