from sentence_transformers import SentenceTransformer


# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = [
    "The cat is sitting on the mat.",
    "A kitten is resting on a rug.",
    "The stock market increased today.",
]


# Convert the sentences into numerical vectors
embeddings = model.encode(sentences)


print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)