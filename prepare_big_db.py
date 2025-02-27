if __name__ == "__main__":
    from lib.EmbeddingProvider import OpenAiEmbeddingProvider
    from lib.DataRepository import DataRepository

    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(model = "text-embedding-3-small"),
        db_path="./data/db/open_ai_small_1000_100_filtered",
        path="./data/r2.0/pdfs",
        name="open_ai_small_1000_100_filtered",
        chunk_size=1_000,
        chunk_overlap=100,
    )
    repo.save_by_file(path = "./data/r2.0/pdfs")