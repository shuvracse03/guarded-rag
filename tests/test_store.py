from app.retrieval.store import VectorStore


def test_store_keeps_working_after_another_handle_resets_collection(tmp_path):
    server_side = VectorStore(str(tmp_path))
    seed_side = VectorStore(str(tmp_path))
    meta = {"tenant": "t", "doc_id": "d"}

    server_side.add_chunks(["d#0"], ["hello"], [[1.0, 0.0]], [meta])
    assert server_side.counts() == (1, 1)

    seed_side.reset()  # what `python -m app.seed --reset` does while a server is running

    assert server_side.counts() == (0, 0)
    server_side.add_chunks(["d#0"], ["hello again"], [[1.0, 0.0]], [meta])
    assert server_side.counts() == (1, 1)
