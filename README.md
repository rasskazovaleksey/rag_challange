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

### Final decision

After discussion with the team we decided to use following approach:

0. Generate the RAG model DBs.
1. Extract key metric from the question: company, industry, metric, currency, etc.
2. Use the extracted key metric to find the most similar question in the database.
3. Use the answer from the most similar question as the answer for the new question.
4. Use the LLM model to generate the answers.
5. Collect the answers and present them to the user.

