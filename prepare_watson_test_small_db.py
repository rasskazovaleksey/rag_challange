if __name__ == "__main__":
    from lib.EmbeddingProvider import WatsonEmbeddingProvider
    from lib.DataRepository import DataRepository

    repo = DataRepository(
        embedding=WatsonEmbeddingProvider(),
        db_path="./data/db/watson_ai_test_150_15_filtered",
        path="./data/r2.0-test/pdfs",
        name="watson_ai_test_150_15_filtered",
        chunk_size=150,
        chunk_overlap=15,
    )
    repo.save_by_file(path = "./data/r2.0-test/pdfs")