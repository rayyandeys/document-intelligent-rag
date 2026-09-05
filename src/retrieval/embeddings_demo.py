from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat is sitting on the mat.",
    "A kitten is resting on a rug.",
    "The stock market increased today.",
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)


# Calculate cosine similarity between two vectors
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    return dot_product / (magnitude_a * magnitude_b)


similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])
similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print("Cat vs Kitten similarity:", similarity_1_2)
print("Cat vs Stock Market similarity:", similarity_1_3)