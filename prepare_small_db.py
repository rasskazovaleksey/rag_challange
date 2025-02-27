if __name__ == "__main__":
    from lib.EmbeddingProvider import OpenAiEmbeddingProvider
    from lib.DataRepository import DataRepository

    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(model = "text-embedding-3-large"),
        db_path="./data/db/open_ai_large_100_10",
        path="./data/r2.0/pdfs",
        name="open_ai_large_100_10",
        chunk_size=1_00,
        chunk_overlap=10,
    )
    repo.save_by_file(path = "./data/r2.0/pdfs")