import rag_store


def test_get_chroma_db_dir_default(monkeypatch):
    monkeypatch.delenv("CHROMA_DB_DIR", raising=False)
    value = rag_store._get_chroma_db_dir()
    assert value == "/chroma"


def test_get_chroma_db_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path))
    value = rag_store._get_chroma_db_dir()
    assert value == str(tmp_path)


def test_get_chroma_collection_name_default(monkeypatch):
    monkeypatch.delenv("CHROMA_COLLECTION_NAME", raising=False)
    value = rag_store._get_chroma_collection_name()
    assert value == "documents"


def test_get_chroma_collection_name_env(monkeypatch):
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "my_collection")
    value = rag_store._get_chroma_collection_name()
    assert value == "my_collection"


def test_get_ollama_base_url_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    value = rag_store._get_ollama_base_url()
    assert value == "http://ollama:11434"


def test_get_ollama_base_url_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1234")
    value = rag_store._get_ollama_base_url()
    assert value == "http://example.com:1234"


def test_get_embedding_model_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    value = rag_store._get_embedding_model()
    # just make sure it's not empty and matches the default
    assert value == "nomic-embed-text"


def test_get_embedding_model_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "my-embed-model")
    value = rag_store._get_embedding_model()
    assert value == "my-embed-model"
