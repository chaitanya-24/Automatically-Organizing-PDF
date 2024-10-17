# PDF Organizer Tool

## Description
The PDF Organizer Tool is a Streamlit application that allows users to upload multiple PDF files, extract abstracts from them, generate embeddings, and organize the PDFs into clusters based on their content. This tool utilizes machine learning techniques to group similar documents, making it easier to manage and retrieve related PDFs.

## Features
- Upload multiple PDF files.
- Extract abstracts from the first page of each PDF.
- Generate embeddings using a pre-trained SentenceTransformer model.
- Cluster PDFs based on their content using KMeans clustering.
- Organize clustered PDFs into separate folders.

## Requirements
- Python 3.7 or higher
- Streamlit
- PyMuPDF
- pandas
- sentence-transformers
- scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Automatically-Organizing-PDF.git
   cd Automatically-Organizing-PDF
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your PDF files using the file uploader.

4. Select the number of clusters you want to create.

5. Click the "Organize PDFs" button to start the clustering process.

6. Once the process is complete, you will see a success message along with the path to the organized PDFs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
