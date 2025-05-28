1) Producer: Uploads PDFs, extracts text from them, and passes it for processing.

class PdfProducerService(processor: PdfProcessor):
+ save(file: str) -> None: 1) same as https://github.com/rasskazovaleksey/rag_challange/blob/main/lib/DataRepository.py 2) emits event to rabiit

1.1) embedding: EmbeddingProvider leave as is but delete Watson code

2) Database: Stores all information in ChromaDB (vector representations).

class ChromaRepository(embedding: EmbeddingProvider)
+ creat(text) -> Vector + Score 
+ read(text) -> Vector + Score

3) PdfProcessor:
   • Processes the text (cleaning, segmentation into fragments). - 1) same as https://github.com/rasskazovaleksey/rag_challange/blob/main/lib/DataRepository.py
   • Creates text embeddings for efficient search. is in ChromaRepository
   • Analyzes queries, including synonym search.
4) Database: Stores all information in ChromaDB (vector representations). - DO NOTHING
 

5) Presenter: - subscribes on rabbit event
   • Performs a vector search in the database upon receiving a query.
   • Retrieves the most relevant text fragments.
   • Passes them to an LLM, which generates a response based on the retrieved data.