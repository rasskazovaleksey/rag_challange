if __name__ == "__main__":
    from lib.EmbeddingProvider import WatsonEmbeddingProvider
    from lib.DataRepository import DataRepository

    repo = DataRepository(
        embedding=WatsonEmbeddingProvider(),
        db_path="./data/db/watson_ai_large_1000_100_filtered",
        path="./data/r2.0/pdfs",
        name="watson_ai_large_1000_100_filtered",
        chunk_size=1000,
        chunk_overlap=100,
    )
    repo.save_by_file(path = "./data/r2.0/pdfs")