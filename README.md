# RAG Challenge

[Enterprise RAG Challenge](https://rag.timetoact.at/)

## Experiments

### RAG via chroma

Fist experiment is based on [rag-tutorial](https://github.com/pixegami/rag-tutorial-v2) from 
[youtube](https://www.youtube.com/watch?v=2TJxpyO3ei4). For text cleaning see, 
[Generate Embeddings using Amazon Bedrock and LangChain Tahir Rauf](https://medium.com/@tahir.rauf/similarity-search-using-langchain-and-bedrock-4140b0ae9c58)

Simple ideas is that we can use the RAG model to generate embeddings for the text and then use the embeddings to 
find similar text. Future work showed what we need to tune some params: prompts, split sizes and embeddings provider,
etc. This stays as baseline. We estimate that this approach gives approximately 60% of the accuracy without tuning.

