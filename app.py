import streamlit as st
import tempfile
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="MultiDoc RAG", page_icon="📚")
st.title("📚 Multi-Document RAG System")
st.markdown("Upload multiple PDFs and ask questions across all of them.")

# Sidebar
with st.sidebar:
    st.header("📖 How It Works")
    st.markdown("""
    1. Upload multiple PDFs
    2. All documents are chunked
    3. Embeddings created for all
    4. Ask a question
    5. System finds best chunks from ANY document
    
    ---
    🆓 100% Free — No API keys
    """)
    st.markdown("### 📊 Document Stats")
    doc_count = st.empty()

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload PDF documents (you can select multiple):",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    chunk_sources = []  # Track which document each chunk came from
    
    st.info(f"📂 Processing {len(uploaded_files)} documents...")
    
    # Process each PDF
    for file_idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        pdf_reader = PyPDF2.PdfReader(tmp_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Chunk this document
        doc_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 100]
        
        # Add to master list with source tracking
        for chunk in doc_chunks:
            all_chunks.append(chunk)
            chunk_sources.append({
                'doc_name': uploaded_file.name,
                'doc_index': file_idx
            })
        
        st.write(f"✅ {uploaded_file.name}: {len(doc_chunks)} chunks")
    
    st.success(f"📦 Total: {len(all_chunks)} chunks from {len(uploaded_files)} documents")
    
    # Create embeddings for all chunks
    with st.spinner("🔢 Creating embeddings for all documents..."):
        chunk_embeddings = model.encode(all_chunks)
    
    st.success("✅ All documents indexed! Ask your questions below.")
    
    # Question input
    question = st.text_input("Ask a question (searches across ALL documents):")
    
    if question:
        with st.spinner("🔍 Searching all documents..."):
            # Get question embedding
            q_embedding = model.encode([question])[0]
            
            # Cosine similarity against all chunks
            similarities = np.dot(chunk_embeddings, q_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
            )
            
            # Get top 5 results
            top_k = 5
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            st.subheader(f"📄 Top {top_k} Results Across All Documents")
            
            for rank, idx in enumerate(top_indices):
                score = similarities[idx]
                source = chunk_sources[idx]
                
                with st.expander(
                    f"#{rank+1} | Score: {score:.2f} | 📄 {source['doc_name']}"
                ):
                    st.markdown(f"**Document:** {source['doc_name']}")
                    st.markdown(f"**Relevance:** {score:.2%}")
                    st.markdown("---")
                    st.write(all_chunks[idx][:800])
            
            # Source summary
            st.markdown("---")
            st.subheader("📊 Source Breakdown")
            
            # Count which documents contributed
            source_counts = {}
            for idx in top_indices:
                doc_name = chunk_sources[idx]['doc_name']
                source_counts[doc_name] = source_counts.get(doc_name, 0) + 1
            
            for doc_name, count in source_counts.items():
                st.metric(f"📄 {doc_name}", f"{count} relevant chunks")