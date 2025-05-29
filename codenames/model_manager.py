"""
Shared Model Manager for Codenames AI
Loads models once and shares them across all agents and tournaments.
"""

import os
import time
from typing import Dict, Any, Optional
from threading import Lock

class ModelManager:
    """
    Singleton class that manages shared models across all agents.
    Ensures each model is loaded only once and shared across all instances.
    """
    
    _instance = None
    _lock = Lock()
    _models: Dict[str, Any] = {}
    _loading_locks: Dict[str, Lock] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            print("Model Manager initialized")
    def get_glove_model(self, model_name: str = "glove-wiki-gigaword-300"):
        """Get shared GloVe model, loading if necessary"""
        return self._get_model(f"glove_{model_name}", self._load_glove_model, model_name)
    
    def get_sbert_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Get shared SBERT model, loading if necessary"""
        return self._get_model(f"sbert_{model_name}", self._load_sbert_model, model_name)
    
    def get_openai_client(self):
        """Get shared OpenAI client, creating if necessary"""
        return self._get_model("openai_client", self._create_openai_client)
    
    def _get_model(self, key: str, loader_func, *args):
        """Generic method to get or load a model with thread safety"""
        # Check if already loaded
        if key in self._models:
            return self._models[key]
        
        # Ensure we have a lock for this model
        if key not in self._loading_locks:
            with self._lock:
                if key not in self._loading_locks:
                    self._loading_locks[key] = Lock()
        
        # Load the model (thread-safe)
        with self._loading_locks[key]:
            # Double-check after acquiring lock
            if key in self._models:
                return self._models[key]
            
            print(f"Loading {key}...")
            start_time = time.time()
            
            try:
                model = loader_func(*args)
                self._models[key] = model
                
                load_time = time.time() - start_time
                print(f"{key} loaded successfully in {load_time:.1f}s")
                
                return model
                
            except Exception as e:
                print(f"Failed to load {key}: {str(e)}")
                raise
    
    def _load_glove_model(self, model_name: str):
        """Load GloVe model using gensim"""
        try:
            from gensim.downloader import load as gensim_load
            return gensim_load(model_name)
        except ImportError:
            raise ImportError("gensim is required for GloVe models. Install with: pip install gensim")
    
    def _load_sbert_model(self, model_name: str):
        """Load Sentence Transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
    def preload_common_models(self):
        """Pre-load commonly used models to avoid delays during tournaments"""
        print("🚀 Pre-loading common models...")
        
        # Load in order of importance/usage
        models_to_load = [
            ("GloVe", lambda: self.get_glove_model()),
            ("SBERT", lambda: self.get_sbert_model()),
        ]
        
        total_start = time.time()
        
        for model_name, loader in models_to_load:
            try:
                loader()
            except Exception as e:
                print(f"Failed to pre-load {model_name}: {e}")
        
        total_time = time.time() - total_start
        print(f"Model pre-loading completed in {total_time:.1f}s")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self._models.keys()),
            'model_count': len(self._models),
            'memory_usage': {}
        }
        
        # Try to get memory usage if possible
        for key, model in self._models.items():
            try:
                if hasattr(model, 'vector_size'):
                    info['memory_usage'][key] = f"Vector size: {model.vector_size}"
                elif hasattr(model, 'get_sentence_embedding_dimension'):
                    info['memory_usage'][key] = f"Embedding dim: {model.get_sentence_embedding_dimension()}"
                else:
                    info['memory_usage'][key] = "Unknown size"
            except:
                info['memory_usage'][key] = "Size unavailable"
        
        return info
    
    def clear_models(self):
        """Clear all loaded models (for testing or memory management)"""
        print("Clearing all models...")
        self._models.clear()
        print("All models cleared")

# Global instance for easy access
model_manager = ModelManager()

# Convenience functions for easy importing
def get_glove_model(model_name: str = "glove-wiki-gigaword-300"):
    """Get shared GloVe model"""
    return model_manager.get_glove_model(model_name)

def get_sbert_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get shared SBERT model"""
    return model_manager.get_sbert_model(model_name)

def get_openai_client():
    """Get shared OpenAI client"""
    return model_manager.get_openai_client()

def preload_models():
    """Pre-load common models"""
    model_manager.preload_common_models()

def get_model_info():
    """Get model information"""
    return model_manager.get_model_info()

# Usage example and testing
if __name__ == "__main__":
    print("Testing Model Manager...")
    
    # Test loading
    print("\n1. Testing GloVe loading:")
    glove1 = get_glove_model()
    glove2 = get_glove_model()  # Should be instant (same instance)
    print(f"Same instance: {glove1 is glove2}")
    
    print("\n2. Testing SBERT loading:")
    sbert1 = get_sbert_model()
    sbert2 = get_sbert_model()  # Should be instant (same instance)
    print(f"Same instance: {sbert1 is sbert2}")
    
    print("\n3. Model info:")
    info = get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nModel Manager test completed!")