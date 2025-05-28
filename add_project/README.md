# How to run the project

## Requirements

Create `tokens.yaml` file in the root directory with the following content:

```yaml
openai: {YOUR_TOKEN}
```

#### RabbiMQ

```bash
docker run --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

#### How to run

```bash
make start
```
#### Run presenter

```bash
make presenter
```

#### Ask question

```json
{
    "text": "What is the total number of employees let go by Pintec Technology Holdings Limited according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
}
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "text": "What is the total number of employees let go by Pintec Technology Holdings Limited according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
}' http://0.0.0.0:8080
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "text": "Did Brave Bison Group plc mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
}' http://0.0.0.0:8080
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "text": "What was the value of Number of hotels at year-end of MGM Resorts International at the end of the period listed in annual report? If data is not available, return N/A.",
  "kind": "number"
}' http://0.0.0.0:8080
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "text": "Which of the companies had the lowest total assets in EUR at the end of the period listed in annual report: \"Datalogic\", \"Terns Pharmaceuticals, Inc.\", \"Incyte Corporation\", \"INMUNE BIO INC.\", \"Duni Group\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
}' http://0.0.0.0:8080
```


```bash
curl -X POST -H "Content-Type: application/json" -d '[
  {
    "text": "For Ziff Davis, Inc., what was the value of Cloud storage capacity (TB) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Liberty Broadband Corporation announce a share buyback plan in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What is the total number of employees let go by Pintec Technology Holdings Limited according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Westwater Resources, Inc. in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did Brave Bison Group plc mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "According to the annual report, what is the Cash flow from operations (in USD) for Sonic Automotive, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Poste Italiane announce any changes to its dividend policy in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the largest single spending of MGM Resorts International on executive compensation in USD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the Gross margin (%) for INMUNE BIO INC. according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did BetMakers Technology Group Ltd mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "For Franklin Covey Co., what was the value of Year-end box office market share (if applicable) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Downer EDI Limited announce a share buyback plan in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the Gross margin (%) for Armadale Capital Plc according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did AA Limited announce any new product launches in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "Did Franklin Covey Co. outline any new ESG initiatives in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the largest single spending of Ocugen, Inc. on executive compensation in AUD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Bionano Genomics, Inc. mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "Did Seiko Epson Corporation announce any changes to its dividend policy in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the value of Number of hotels at year-end of MGM Resorts International at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What is the total number of employees let go by NZME Limited according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Incyte Corporation, what was the value of Clinical trial sites operating at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Aurora Innovation, Inc., what was the value of Number of patents at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest total assets in EUR at the end of the period listed in annual report: \"Datalogic\", \"Terns Pharmaceuticals, Inc.\", \"Incyte Corporation\", \"INMUNE BIO INC.\", \"Duni Group\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "What is the total number of employees let go by Downer EDI Limited according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest total revenue in EUR at the end of the period listed in annual report: \"Atreca, Inc.\", \"Poste Italiane\", \"Datalogic\", \"NuCana plc\", \"RWE AG\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "What was the value of Total power generation capacity (MW) of Elixir Energy Limited at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the value of Number of active pharmaceutical patents of Kiniksa Pharmaceuticals, Ltd. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the value of Total deposits at year-end of CoreCard Corporation at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For HCA Healthcare, Inc., what was the value of Outstanding insurance claims (if applicable) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Datalogic in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did Incitec Pivot Limited mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "For Franklin Covey Co., what was the value of Number of active licensing deals at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "According to the annual report, what is the Cash flow from operations (in USD) for Wheeler Real Estate Investment Trust, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Incitec Pivot Limited announce any changes to its dividend policy in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the largest single spending of archTIS Limited on executive compensation in USD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Guaranty Bancshares, Inc. announce any new product launches in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "According to the annual report, what is the Cash flow from operations (in GBP) for AA Limited  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Peako Limited, what was the value of Cloud storage capacity (TB) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "According to the annual report, what is the Total revenue (in USD) for Medallion Financial Corp.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did AA Limited report any changes to its capital structure? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What is the total number of employees let go by KP Tissue Inc. according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest total revenue in EUR at the end of the period listed in annual report: \"Atreca, Inc.\", \"Poste Italiane\", \"Datalogic\", \"Duni Group\", \"Incyte Corporation\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "Which leadership positions changed at Blue Apron Holdings, Inc. in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "What was the Dividend per share (in USD) for Ritchie Bros. Auctioneers Incorporated according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What are the names of new products launched by Albany International Corp. as mentioned in the annual report?",
    "kind": "names"
  },
  {
    "text": "For Sonic Automotive, Inc., what was the value of Number of hybrid models available at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did ACRES Commercial Realty Corp. outline any new ESG initiatives in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the value of Generic product count of Kiniksa Pharmaceuticals, Ltd. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the value of Number of fulfillment centers at year-end of 1-800-FLOWERS.COM, INC. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the largest single spending of Kiniksa Pharmaceuticals, Ltd. on executive compensation in USD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Origin Bancorp, Inc., what was the value of Total assets on balance sheet at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Ritchie Bros. Auctioneers Incorporated mention any ongoing litigation or regulatory inquiries? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What is the total number of employees let go by Commerzbank according to the annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest total assets in EUR at the end of the period listed in annual report: \"Poste Italiane\", \"NuCana plc\", \"Incyte Corporation\", \"INMUNE BIO INC.\", \"Atreca, Inc.\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "For HCA Healthcare, Inc., what was the value of Number of managed clinics at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For RWE AG, what was the value of Number of facilities at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest net income in EUR at the end of the period listed in annual report: \"Atreca, Inc.\", \"INMUNE BIO INC.\", \"Datalogic\", \"NuCana plc\", \"RWE AG\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "For Albany International Corp., what was the value of R&D spending on advanced programs at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Rectifier Technologies Ltd, what was the value of Number of patents at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Albany International Corp., what was the value of Year-end patent portfolio (aerospace tech) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest net income in EUR at the end of the period listed in annual report: \"Datalogic\", \"NuCana plc\", \"Duni Group\", \"Playtech plc\", \"Atreca, Inc.\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "For SThree plc, what was the value of End-of-year total headcount at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which of the companies had the lowest total assets in EUR at the end of the period listed in annual report: \"Playtech plc\", \"Datalogic\", \"Duni Group\", \"Poste Italiane\", \"Incyte Corporation\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "For HCA Healthcare, Inc., what was the value of Number of healthcare professionals on staff at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For SIG plc, what was the value of Number of stores at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Kelly Partners Group Holdings Limited in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did Trinity Place Holdings Inc. mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "For FNCB Bancorp, Inc., what was the value of Non-performing loan ratio (NPL) at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Elixir Energy Limited outline any new ESG initiatives in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the value of Year-end user base of archTIS Limited at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the largest single spending of MainStreet Bancshares, Inc. on executive compensation in USD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the Capital expenditures (in USD) for Structural Monitoring Systems Plc according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the Capital expenditures (in EUR) for INMUNE BIO INC. according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What is the name of the last product launched by 1-800-FLOWERS.COM, INC. as mentioned in the annual report?",
    "kind": "name"
  },
  {
    "text": "For Peako Limited, what was the value of Year-end customer base at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "According to the annual report, what is the Cash flow from operations (in USD) for FNCB Bancorp, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Peako Limited, what was the value of Total expensed R&D expenditure at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Empire Company Limited announce any changes to its dividend policy in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "Which leadership positions changed at Duni Group in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did SIG plc mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "For Pintec Technology Holdings Limited, what was the value of End-of-year net interest margin (NIM) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For AA Limited, what was the value of Fleet size (vehicles) at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did HCA Healthcare, Inc. announce any changes to its dividend policy in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "Which of the companies had the lowest total assets in EUR at the end of the period listed in annual report: \"Incyte Corporation\", \"INMUNE BIO INC.\", \"Datalogic\", \"Terns Pharmaceuticals, Inc.\", \"RWE AG\"? If data for the company is not available, exclude it from the comparison. If only one company is left, return this company.",
    "kind": "name"
  },
  {
    "text": "What was the value of E-commerce active customer accounts of Mosaic Brands Limited at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the largest single spending of Toshiba Corporation on executive compensation in AUD? If data is not available in this currency, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Sonic Automotive, Inc., what was the value of Year-end fleet average CO₂ emissions at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Did Wheeler Real Estate Investment Trust, Inc. report any changes to its capital structure? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "For Atreca, Inc., what was the value of Number of managed clinics at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Crombie REIT in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did Mosaic Brands Limited mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "Did Incitec Pivot Limited detail any restructuring plans in the latest filing? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "What was the value of Number of active software licenses of Rapid7 at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Wheeler Real Estate Investment Trust, Inc. in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "Did Aptevo Therapeutics Inc. mention any mergers or acquisitions in the annual report? If there is no mention, return False.",
    "kind": "boolean"
  },
  {
    "text": "According to the annual report, what is the Cash flow from operations (in GBP) for James Halstead plc  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "What was the value of End-of-year tech staff headcount of archTIS Limited at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "For Westwater Resources, Inc., what was the value of Percentage of renewable energy capacity at the end of the period listed in annual report? If data is not available, return 'N/A'.",
    "kind": "number"
  },
  {
    "text": "Which leadership positions changed at Origin Bancorp, Inc. in the reporting period? If data is not available, return 'N/A'. Give me the title of the position.",
    "kind": "names"
  },
  {
    "text": "What was the Gross margin (%) for Ritchie Bros. Auctioneers Incorporated according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
    "kind": "number"
  }
]' http://0.0.0.0:8080
```


# Project Overview

The project implements a RAG (Retrieval Augmented Generation) pipeline to answer questions based on a collection of PDF documents. It utilizes RabbitMQ for message queuing between services, ChromaDB as a vector store for document embeddings, and an OpenAI LLM for answer generation.

**1. PDF Ingestion and Processing:**

*   **`PdfProducerService.py` (triggered by `make start`):**
    *   Monitors a specified directory (default: `data/r2.0/pdfs`) for PDF files.
    *   Reads each PDF and sends its filename and binary content to a RabbitMQ queue (`pdf_queue`) using `PdfSender`.

*   **`PdfProccesorService.py` (triggered by `make start`):**
    *   Consumes PDF data (filename and bytes) from the `pdf_queue` via `PdfReceiver`.
    *   For each PDF:
        *   Loads the document content using `PyPDFLoader`.
        *   Splits the document into smaller text chunks using `RecursiveCharacterTextSplitter`.
        *   Assigns unique IDs to each chunk based on filename, page, and chunk index.
        *   Cleans the text content of each chunk (lowercase, remove URLs, HTML, punctuation, stopwords).
        *   Uses `PdfRepo.py` to store these processed chunks.

*   **`PdfRepo.py` (used by `PdfProccesorService.py` and `ExperimentPipeline.py` via `DataRepository`):**
    *   Manages interactions with the ChromaDB vector database.
    *   When creating entries, it generates embeddings for the text chunks using an `EmbeddingProvider` (specifically `OpenAiEmbeddingProvider` with a model like `text-embedding-3-small`).
    *   Stores the text chunks along with their metadata (including SHA1 of the source PDF) and vector embeddings in ChromaDB, avoiding duplicates based on chunk ID.
    *   Provides a `query` method to perform similarity searches in ChromaDB.

**2. Question Handling and Answering:**

*   **`presenter.py` (triggered by `make present`):**
    *   Runs an HTTP server (default: `0.0.0.0:8080`).
    *   Listens for POST requests on the root path (`/`).
    *   Expects a JSON payload containing either a single question object or a list of question objects. Each question object should have "text" (the question itself) and "kind" (e.g., "number", "boolean", "name").
    *   Forwards the received question(s) to a RabbitMQ queue (`json_queue`) using `JsonSender`.

*   **`consumer.py` - `JsonProcessor` (triggered by `make start`):**
    *   Consumes JSON question objects from the `json_queue` via `JsonReceiver`.
    *   For each question, it calls the `run` method of the `ExperimentPipeline.py`.
    *   The `handle_json` method in `consumer.py` is responsible for processing the question and (as per previous implementation, currently commented out) writing the question and its answer to `answer.json`.

*   **`ExperimentPipeline.py` (used by `consumer.py`):**
    *   This is the core RAG logic. For a given question:
        *   **Extracts Information:** Uses `QuestionExtractor` to parse the question text, identifying key metrics, company names. It determines the SHA1 identifiers for the relevant company PDF(s) by looking them up in `data/r2.0/subset.json`.
        *   **Searches Database:** Queries ChromaDB (via `DataRepository`, which uses `PdfRepo`) using the original question and extracted metrics to find the most relevant text chunks from the PDFs associated with the identified SHA1(s).
        *   **Filters Candidates:** Ranks and filters the retrieved chunks based on relevance scores and page occurrences to select the best candidate pages.
        *   **Reads PDF Context:** Loads the full text of the selected candidate pages from the original PDF files stored locally.
        *   **Generates Answer:** Constructs a prompt using the original question and the retrieved page content. This prompt is sent to an LLM (configured as `OpenAIAgent`, using OpenAI's API) to generate a final answer. The specific prompt template used depends on the "kind" of the question (e.g., `number_prompt.txt`, `boolean_prompt.txt`).
        *   The pipeline returns a dictionary containing the original question, the SHA1(s) of the source document(s), and the generated answer. This result is then processed by `consumer.py`.

**Key Technologies & Libraries:**

*   **Python:** Core programming language.
*   **RabbitMQ:** Message broker for asynchronous communication between services (`PdfProducerService`, `PdfProccesorService`, `presenter`, `consumer`).
*   **ChromaDB:** Vector database for storing and searching text embeddings (`PdfRepo`).
*   **Langchain:** Framework used for document loading (`PyPDFLoader`), text splitting (`RecursiveCharacterTextSplitter`), and interacting with embedding models (`OpenAIEmbeddings`) and LLMs.
*   **OpenAI API:** Used for generating text embeddings (`OpenAiEmbeddingProvider`) and for the final answer generation by the LLM (`OpenAIAgent`).
*   **NLTK:** Used for text processing tasks like stopword removal (`PdfProccesorService`).
*   **Makefile:** For easy starting and stopping of project services.
*   **HTTP Server (http.server):** Used in `presenter.py` to receive questions.