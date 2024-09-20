# Import necessary libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Step 1: Load the CSV file
file_path = 'data/customers-100.csv'
df = pd.read_csv(file_path)

# Step 2: Combine relevant fields into a single text column for embedding
df['combined_text'] = df['First Name'] + ' ' + df['Last Name'] + ', works for ' + df['Company'] + '. Located in ' + df['City'] + ', ' + df['Country'] + '. Contact: ' + df['Email'] + '.'

# Step 3: Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model from Hugging Face

# Step 4: Generate embeddings for the combined text
doc_texts = df['combined_text'].tolist()  # Convert the combined text column to a list
text_vectors = model.encode(doc_texts, convert_to_tensor=True)

# Step 5: Move the tensor to CPU and convert to a numpy array
vector_matrix = text_vectors.cpu().numpy()

# Step 6: Create the FAISS index (Using L2 distance-based search)
index = faiss.IndexFlatL2(vector_matrix.shape[1])

# Step 7: Add the vectors to the FAISS index
index.add(vector_matrix)

# Step 8: Define a function to search for similar documents
def search_faiss(query, k=5):
    # Move query vector to CPU
    query_vector = model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_vector, k)  # k = number of results to return
    
    # Fetch the most relevant text chunks based on the indices returned by FAISS
    results = [doc_texts[i] for i in I[0]]
    
    return results

# Example query
query = "Which company does Sheryl Baxter work for?"
top_results = search_faiss(query, k=5)

# Display the top results
for i, result in enumerate(top_results, 1):
    print(f"Result {i}: {result}")
