import PyPDF2
from sentence-transformers import SentenceTransformer  # Correct import
import faiss
import numpy as np
import openai

# Step 1: Extract text from PDF files
def extract_text_from_pdfs(pdf_files):
    texts = []
    for file in pdf_files:
        with open(file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None if extraction fails
            texts.append(text)
    return texts

# Step 2: Embed text using Sentence Transformers
def embed_texts(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, convert_to_tensor=True)

# Step 3: Create a FAISS index for fast retrieval
def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(np.array(embeddings))  # Add embeddings to the index
    return index

# Step 4: Retrieve relevant documents
def retrieve_documents(query, index, texts, model, k=3):
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().numpy(), k)  # D is distances, I is indices
    return [texts[i] for i in I[0]]

# Step 5: Generate a response with GPT-3.5 Turbo
def generate_response(relevant_texts, query):
    prompt = f"Based on the following documents, answer the question:\n\n{query}\n\nDocuments:\n{relevant_texts}\n\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

# Main Workflow
def main(pdf_files, user_query):
    texts = extract_text_from_pdfs(pdf_files)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_texts(texts)
    index = create_index(embeddings)

    relevant_documents = retrieve_documents(user_query, index, texts, model)
    answer = generate_response(relevant_documents, user_query)

    print("Answer:", answer)

# Example usage
if __name__ == "__main__":
    # List of your PDF files
    pdf_files = ['file1.pdf', 'file2.pdf']  # Add your PDF files here
    user_query = "What is the main topic discussed in the documents?"  # Your query here

    # Ensure you set your OpenAI API key
    openai.api_key = 'YOUR_API_KEY'  # Replace with your OpenAI API key

    main(pdf_files, user_query)
