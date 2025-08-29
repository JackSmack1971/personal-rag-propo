from sentence_transformers import SentenceTransformer

_model_cache = {}

def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]
