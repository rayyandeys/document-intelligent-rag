"""MiniLM embeddings with an optional, PyTorch-free hosting runtime."""
import os
from typing import List, Dict

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        elif self.backend == "onnx":
            if model_name != "sentence-transformers/all-MiniLM-L6-v2":
                raise ValueError("The ONNX pooling configuration supports all-MiniLM-L6-v2 only")
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            revision = os.getenv("MINILM_REVISION", "main")
            self.tokenizer = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json", revision=revision))
            self.tokenizer.enable_truncation(max_length=256)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            # Avoid extra optimized graph copies during startup on small hosts.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.session = ort.InferenceSession(
                hf_hub_download(model_name, "onnx/model.onnx", revision=revision),
                sess_options=options, providers=["CPUExecutionProvider"],
            )
        else:
            raise ValueError("Unknown EMBEDDING_BACKEND")

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if self.backend == "sentence_transformers":
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vectors = []
        # One chunk at a time bounds temporary attention tensors.
        for text in texts:
            encoded = self.tokenizer.encode_batch([text.strip()])
            inputs = {
                "input_ids": np.asarray([x.ids for x in encoded], dtype=np.int64),
                "attention_mask": np.asarray([x.attention_mask for x in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([x.type_ids for x in encoded], dtype=np.int64),
            }
            names = {item.name for item in self.session.get_inputs()}
            token_vectors = self.session.run(None, {k: v for k, v in inputs.items() if k in names})[0]
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            pooled = (token_vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.append(pooled[0])
        return np.asarray(vectors, dtype=np.float32)

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        return self._encode([chunk["text"] for chunk in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query])[0]
