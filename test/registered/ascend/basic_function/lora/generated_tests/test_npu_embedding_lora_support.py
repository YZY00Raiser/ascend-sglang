"""
Test NPU Embedding LoRA support - 验证NPU上Embedding模型的LoRA功能

Tests LoRA support for embedding models on Ascend NPU:
1. Engine.encode() with LoRA validation
2. EmbeddingReqInput LoRA fields
3. Batch normalization and indexing
"""

import os
import sys
import pytest
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_embedding,
    call_sglang_load_adapter,
)

# Test configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Embedding model with LoRA support
ADAPTER_URL = "https://huggingface.co/sglang/bge-lora-test"

TEXTS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning involves training agents through rewards and penalties.",
]

SIMILAR_PAIRS = [
    ("Machine learning is a subset of artificial intelligence.",
     "ML is a branch of AI that enables computers to learn from data."),
    ("Deep learning uses neural networks.",
     "Neural networks with multiple layers are used in deep learning."),
]

DISSIMILAR_PAIRS = [
    ("Machine learning is a subset of artificial intelligence.",
     "The capital of France is Paris."),
    ("Deep learning uses neural networks.",
     "Pizza is a popular Italian dish."),
]


class TestNpuEmbeddingLoRA:
    """Test Embedding LoRA support on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with embedding model."""
        print("\n[SGLang] Starting server for embedding LoRA testing...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "is_embedding_model": True,
            "lora_paths": f"{{'embedding-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def _get_embeddings(self, texts: List[str], lora_path: str = None) -> np.ndarray:
        """Get embeddings for a list of texts."""
        response = call_sglang_embedding(
            texts=texts,
            lora_path=lora_path,
            normalize=True,
        )
        
        embeddings = []
        for item in response.get("data", []):
            embedding = item.get("embedding", [])
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_basic_embedding_with_lora(self, server_process):
        """Test basic embedding generation with LoRA."""
        print("\n[Test] Testing basic embedding with LoRA...")
        
        # Get embeddings with LoRA
        print("  Getting embeddings with LoRA...")
        embeddings_with_lora = self._get_embeddings(TEXTS[:3], lora_path="embedding-lora")
        
        # Get embeddings without LoRA
        print("  Getting embeddings without LoRA...")
        embeddings_without_lora = self._get_embeddings(TEXTS[:3], lora_path=None)
        
        # Verify dimensions
        assert embeddings_with_lora.shape == embeddings_without_lora.shape, \
            "Embeddings should have same dimensions"
        
        # Verify embeddings are different with LoRA
        diff = np.linalg.norm(embeddings_with_lora - embeddings_without_lora)
        print(f"  Embedding difference (L2 norm): {diff:.4f}")
        
        assert diff > 0.1, "LoRA should produce different embeddings"
        
        print("  ✅ Basic embedding with LoRA works correctly")

    def test_embedding_similarity_semantic(self, server_process):
        """Test that similar texts have higher similarity with LoRA embeddings."""
        print("\n[Test] Testing embedding semantic similarity with LoRA...")
        
        # Get embeddings for similar and dissimilar pairs
        similar_texts = [pair[0] for pair in SIMILAR_PAIRS] + [pair[1] for pair in SIMILAR_PAIRS]
        dissimilar_texts = [pair[0] for pair in DISSIMILAR_PAIRS] + [pair[1] for pair in DISSIMILAR_PAIRS]
        
        all_texts = similar_texts + dissimilar_texts
        embeddings = self._get_embeddings(all_texts, lora_path="embedding-lora")
        
        # Compute similarities for similar pairs
        similar_scores = []
        for i, (text1, text2) in enumerate(SIMILAR_PAIRS):
            idx1 = similar_texts.index(text1)
            idx2 = similar_texts.index(text2)
            sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
            similar_scores.append(sim)
            print(f"  Similar pair {i+1}: {sim:.4f}")
        
        # Compute similarities for dissimilar pairs
        dissimilar_scores = []
        for i, (text1, text2) in enumerate(DISSIMILAR_PAIRS):
            idx1 = len(similar_texts) + dissimilar_texts.index(text1)
            idx2 = len(similar_texts) + dissimilar_texts.index(text2)
            sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
            dissimilar_scores.append(sim)
            print(f"  Dissimilar pair {i+1}: {sim:.4f}")
        
        # Similar pairs should have higher similarity than dissimilar pairs
        avg_similar = sum(similar_scores) / len(similar_scores)
        avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores)
        
        print(f"  Average similar: {avg_similar:.4f}")
        print(f"  Average dissimilar: {avg_dissimilar:.4f}")
        
        assert avg_similar > avg_dissimilar, \
            f"Similar pairs should have higher similarity (similar={avg_similar:.4f}, dissimilar={avg_dissimilar:.4f})"
        
        print("  ✅ Embedding semantic similarity works correctly")

    def test_batch_embedding_with_lora(self, server_process):
        """Test batch embedding with LoRA."""
        print("\n[Test] Testing batch embedding with LoRA...")
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            texts = TEXTS[:batch_size]
            
            # Batch embedding with LoRA
            batch_embeddings = self._get_embeddings(texts, lora_path="embedding-lora")
            
            # Individual embeddings with LoRA
            individual_embeddings = []
            for text in texts:
                emb = self._get_embeddings([text], lora_path="embedding-lora")
                individual_embeddings.append(emb[0])
            
            # Compare batch and individual results
            for i in range(batch_size):
                diff = np.linalg.norm(batch_embeddings[i] - individual_embeddings[i])
                assert diff < 1e-5, f"Batch and individual embeddings should match for item {i}"
            
            print(f"    ✅ Batch size {batch_size} verified")
        
        print("  ✅ Batch embedding with LoRA works correctly")

    def test_embedding_normalization(self, server_process):
        """Test that embeddings are properly normalized."""
        print("\n[Test] Testing embedding normalization...")
        
        embeddings = self._get_embeddings(TEXTS, lora_path="embedding-lora")
        
        # Check L2 norm of each embedding
        norms = np.linalg.norm(embeddings, axis=1)
        
        print(f"  Embedding norms: {norms}")
        
        # Embeddings should be normalized (close to 1.0)
        for i, norm in enumerate(norms):
            assert 0.99 <= norm <= 1.01, f"Embedding {i} should be normalized (norm={norm:.4f})"
        
        print("  ✅ Embedding normalization verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
