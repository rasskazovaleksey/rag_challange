if __name__ == "__main__":
    from lib.EmbeddingProvider import WatsonEmbeddingProvider
    from lib.DataRepository import DataRepository

    repo = DataRepository(
        embedding=WatsonEmbeddingProvider(),
        db_path="./data/db/watson_ai_large_100_10_filtered",
        path="./data/r2.0/pdfs",
        name="watson_ai_large_100_10_filtered",
        chunk_size=100,
        chunk_overlap=10,
    )
    repo.save_by_file(path = "./data/r2.0/pdfs")