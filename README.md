#  Semantic Search
`Semantic search` seeks to improve search accuracy by understanding the `semantic meaning` of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

## Background
The idea behind semantic search is to embed all entries in your corpus, whether they be sentences, paragraphs, or documents, into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from your corpus are found. These entries should have a high semantic similarity with the query.

# Tools Used
The project is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| NumPy | Numerical computing library |
| Pandas | Data manipulation library |
| Matplotlib | Data visualization library |
| Sklearn | Machine learning library |
| pytorch | Open-source machine learning framework |
| Transformers | Hugging Face package contains state-of-the-art Natural Language Processing models |
| Datasets | Hugging Face package contains datasets |
| Senetnce Transformer | module for accessing, using, and training state-of-the-art text and image embedding models.   |
| langchain |Orchestration framework to work with LLMs  |
| pinecone |A vector database designed to store, manage, and search large-scale vector embeddings efficiently.  |

# Dataset
1. [Quora-QuAD on Hugging Face](https://huggingface.co/datasets/quora-competitions/quora)

* `questions`: a dictionary feature containing:
* `id`: a int32 feature.
* `questions`: a dictionary feature containing:
* `text`: a string feature.
* `questions`: a dictionary feature containing:
* `is_duplicate`: a bool feature.

2. [Raw_memes](https://raw.githubusercontent.com/bhavya-giri/retrieving-memes/refs/heads/main/data/raw_memes.json)
* the dataset seems to contain *5,823* meme entries, each with the following attributes:

* `Image Captions`: Text that describes the content or context of the meme image.
* `Title`: The title or name given to the meme.
* `URL of Meme`: A direct link to the meme image or its source.
* `Metaphors`: Possibly a field containing figurative language or metaphorical references used in the meme.
* `Meme Captions`: The main text or description accompanying the meme image, which may convey humor or context.

# Methodology
## Dataset Preparation

1. Quora dataset : See [hybrid Search pinecone notebook](https://colab.research.google.com/drive/1Gu2Qx6tHM_9tE8_-Zz-lxFXfnuWgMwDv#scrollTo=2zTHceaEUyJT)

For the Quora dataset, I focused only on the "questions" column. The steps I followed are:

1. `Extracted the Questions`: I looped through the "questions" column.
2. `Removed Duplicates`: I filtered out any duplicate questions to ensure data quality.
3. `Selected a Subset`: From the remaining questions, I picked around 200 unique ones to work with.
4. `Embedded the Questions`: These 200 questions were then embedded into vector representations.
5. `Vector Database`: The embedded questions were added to a vector database to enable hybrid search.
By using `hybrid search`, I aim to combine the benefits of both semantic search (through vector similarity) and traditional keyword-based search for better result relevance.

2.Raw memes dataset : See [Semantic-Search-Senetence trasnfomer notebook](https://colab.research.google.com/drive/1jGB0lHqLyQr2gJYHUx5z33r2gEHhpync#scrollTo=8dV9xiGYto95) 
1. `Combine Text Fields`: I combined all the text-related fields **(title, image caption, meme captions, and metaphors)** into a single column. This helps consolidate the various textual components of the meme into one unified representation.

2. `Separate URL Column`: I kept the URL of the meme in a separate column. This allows me to maintain a reference to the original image while focusing on the textual content for analysis.


# Search methods used
## 1- Hybrid Search: 

`Reciprocal Rank Fusion` is an ensemble-based ranking method used in information retrieval
to combine ranked lists from multiple retrieval models or sources.

How RRF Works:
RRF assigns a score to each document based on its rank in the ranked lists. The fused score for
a document is computed using the formula:

    RRF(d) = sum(1 / (k + r_k(d))) for k = 1 to n

Where:
- `n`: The number of ranked lists being fused.
- `r_k(d)`: The rank of document d in the k-th ranked list (1-based rank).
- `k`: A small constant **(usually set to 60)** to avoid division by zero and reduce the impact of low ranks.

Key Points:
- Documents appearing higher in the rankings of multiple lists get higher fused scores.
- Lower-ranked documents contribute less to the overall score due to the reciprocal nature of the scoring function.
- The parameter k controls how steeply the contribution of lower ranks diminishes.

Example:

Input Ranked Lists:
1. `BM25`:
   - Doc1: Rank 1
   - Doc2: Rank 2
   - Doc3: Rank 3
2. `Dense Retrieval`:
   - Doc2: Rank 1
   - Doc1: Rank 2
   - Doc4: Rank 3

Compute RRF Scores (k = 60):
- For Doc1:
  RRF = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
- For Doc2:
  RRF = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
- For Doc3:
  RRF = 1/(60+3) = 0.01587
- For Doc4:
  RRF = 1/(60+3) = 0.01587

Fused Ranking:
1. Doc1 and Doc2 (tie): 0.03252
2. Doc3 and Doc4 (tie): 0.01587

Advantages of RRF:
1. Simplicity: RRF is computationally inexpensive and straightforward to implement.
2. Robustness: It works well even when the models being combined have widely varying performance.
3. No Need for Parameter Tuning: The small constant k is fixed and doesn't require significant optimization.


## 2. Dense Retrival
**util.semantic_search**

This function performs a cosine similarity search between a list of query embeddings
and a list of corpus embeddings. It can be used for Information Retrieval / Semantic
Search for corpora up to about 1 Million entries.

Parameters:
- `query_embeddings` (Tensor): A 2-dimensional tensor with the query embeddings.
- `corpus_embeddings` (Tensor): A 2-dimensional tensor with the corpus embeddings.
- `query_chunk_size` (int, optional): Process 100 queries simultaneously.
  Increasing that value increases the speed but requires more memory. Defaults to 100.
- `corpus_chunk_size` (int, optional): Scans the corpus 100k entries at a time.
  Increasing that value increases the speed but requires more memory. Defaults to 500000.
- `top_k` (int, optional): Retrieve top k matching entries. Defaults to 10.
- `score_function` (Callable[[Tensor, Tensor], Tensor], optional):
  Function for computing scores. By default, cosine similarity.

## 3. BM25: Best Matching 25

BM25 is a ranking function widely used in search engines and information retrieval to score the relevance of documents to a given query. It improves on traditional methods like TF-IDF by incorporating mechanisms for term saturation and document length normalization.

### Key Features

| **Feature**                     | **Description**                                                                 |
|----------------------------------|---------------------------------------------------------------------------------|
| **Term Frequency (TF)**          | Measures how often a term appears in a document, adjusted for saturation.      |
| **Inverse Document Frequency**   | Reduces the importance of common terms and emphasizes rare, discriminative ones.|
| **Document Length Normalization**| Normalizes scores to avoid bias toward longer documents.                        |

### BM25 Formula

The relevance score of a document \( D \) for a query \( Q \) is calculated as:

\[
\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}})}
\]

Where:
- \( \text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5} \)
  - \( N \): Total number of documents.
  - \( n_t \): Number of documents containing term \( t \).
- \( \text{TF}(t, D) \): Frequency of term \( t \) in document \( D \).
- \( |D| \): Length of document \( D \).
- \( \text{avgD} \): Average document length.
- \( k_1 \): Controls term frequency saturation (default: 1.2–2.0).
- \( b \): Controls document length normalization (default: 0.75).

### Parameters

| **Parameter** | **Effect**                                                                               |
|---------------|-----------------------------------------------------------------------------------------|
| **`k1`**      | Controls the impact of term frequency (higher values increase term saturation).         |
| **`b`**       | Adjusts document length normalization (0 = no normalization, 1 = full normalization). |

### Advantages

- **Effective Ranking**: Balances term frequency, rarity, and document length.
- **Customizable**: Tunable parameters \( k_1 \) and \( b \) for different scenarios.
- **Simplicity**: Straightforward to implement and interpret.

### Use Cases

- **Search Engines**: Ranking web pages or documents based on query relevance.
- **Information Retrieval**: Recommender systems and hybrid search.
- **Text Mining**: Locating relevant information in large text datasets.

## Summary
BM25 is a powerful ranking function designed for efficient and effective document retrieval. It combines:
1. **TF-IDF** to weigh terms by importance and frequency.
2. **Document length normalization** to handle documents of varying sizes.

Its tunable parameters make it adaptable for various applications, from search engines to large-scale text mining.


## Conclusion
# Comparison: `all-mpnet-base-v2` vs `all-MiniLM-L6-v2`

This table compares key attributes of the `all-mpnet-base-v2` and `all-MiniLM-L6-v2` embedding models from SentenceTransformers:

| **Attribute**          | **`all-mpnet-base-v2`**                              | **`all-MiniLM-L6-v2`**                     |
|-------------------------|------------------------------------------------------|--------------------------------------------|
| **Embedding Dimension** | 768                                                  | 384                                        |
| **Context Length**      | 512 tokens                                           | 384 tokens                                 |
| **Model Size**          | Large (110M parameters)                              | Smaller (22M parameters)                   |
| **Speed**               | Slower                                               | Faster                                     |
| **Use Case**            | High-quality embeddings for detailed representations | Lightweight, fast embeddings for real-time or large-scale tasks |

### Summary

- **`all-mpnet-base-v2`**:
  - Embedding size of 768.
  - Supports up to 512 tokens.
  - Suitable for tasks requiring high-quality embeddings, such as semantic search, clustering, or advanced NLP tasks.

- **`all-MiniLM-L6-v2`**:
  - Embedding size of 384.
  - Supports up to 384 tokens.
  - Ideal for real-time or large-scale applications where speed and efficiency are crucial.

Choose the model based on your application's requirements for embedding quality, speed, and scale.
---
# Difference Between `util.cos_sim` and `util.semantic_search`

This table compares the key differences between the `util.cos_sim` and `util.semantic_search` methods in SentenceTransformers:

| **Aspect**            | **`util.cos_sim`**                           | **`util.semantic_search`**                        |
|------------------------|---------------------------------------------|--------------------------------------------------|
| **Primary Function**   | Computes cosine similarity between embeddings. | Ranks documents by relevance to queries.         |
| **Output**             | Similarity matrix (`n x m`).                | Ranked list of top matches for each query.       |
| **Use Case**           | Comparing embeddings directly.              | Retrieving top-k relevant results.               |
| **Scalability**        | Lightweight, direct computation.            | Optimized for large-scale search.                |

### Summary

- **`util.cos_sim`**:
  - Computes a similarity matrix directly between two sets of embeddings.
  - Useful for small-scale comparisons or analysis.

- **`util.semantic_search`**:
  - Optimized for searching and ranking large datasets.
  - Returns the most relevant matches for a given query with scores and IDs.
# comparison of search types
---
| **Feature**             | **Hybrid Search**                                       | **Semantic Search**                                   | **Keyword Search**                                   |
|-------------------------|---------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| **Search Approach**      | Combines traditional keyword search and semantic search (using vector-based retrieval). | Focuses on the meaning of the query and documents using embeddings. | Relies on exact matching of query terms with document terms. |
| **Representation**       | Uses both traditional inverted index and vector space models (e.g., TF-IDF + embeddings). | Uses dense vector representations (e.g., word embeddings, BERT). | Uses an inverted index for term-based matching.      |
| **Accuracy**             | High accuracy for various types of queries, combining strengths of both methods. | High accuracy for contextually related queries, even if exact words don't match. | Can be less accurate, especially for queries with synonyms or variations. |
| **Query Handling**       | Handles both exact and semantically related queries.  | Handles queries based on meaning, even if word choices differ. | Handles queries based on exact keywords.             |
| **Scalability**          | Requires more computational resources (multiple models). | Computationally intensive (due to embeddings).        | Highly scalable and efficient.                       |
| **Speed**                | Slower than keyword search, but more flexible and accurate. | Typically slower due to complex models (e.g., deep learning). | Fast, especially with optimized indexing techniques. |
| **Use Cases**            | Used when a combination of accuracy and speed is required, e.g., in e-commerce, hybrid search engines. | Used for nuanced, meaning-based queries, such as Q&A systems, chatbots, or recommendation engines. | Best for situations with well-defined keywords or when exact matches are needed, e.g., basic search engines. |
| **Strengths**            | Best of both worlds—combines accuracy and flexibility. | Excels at understanding context and semantic meaning. | Fast and efficient for exact term matching.          |
| **Weaknesses**           | More complex to implement and maintain.                | Computationally heavy, may require advanced hardware or cloud services. | Limited to exact keyword matches, struggles with synonyms and variations. |
| **Example**              | A search engine returning results based on both keywords and related semantic context (e.g., e-commerce platform). | A chatbot answering questions about the meaning of a word, even with different phrasing. | A search engine finding results for the exact terms entered by the user (e.g., Google search). |


# References

1. **Semantic Search with Sentence-Transformers**:  
   For a complete guide and examples of semantic search using Sentence-Transformers, 
   visit the official documentation here: 
   [Semantic Search Guide](https://www.sbert.net/examples/applications/semantic-search/README.html)

2. **all-mpnet-base-v2**:  
   This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.
   You can find more information and download it from the Hugging Face model hub:
   [sentence-transformers/all-mpnet-base-v2all-mini-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

3. **Building Simple Semantic Search using Sentence Transformers blog post** 
[blog post](https://medium.com/@bhavyagiri/building-simple-semantic-search-using-sentence-transformers-48ba17bf9b01)


4. **SentenceTransformers Documentation**:  
   For more information on using SentenceTransformers, refer to the official documentation:
   [SentenceTransformers](https://www.sbert.net/)

5. **Hybrid Search with Pinecone**:  
   Learn about hybrid search in Pinecone, combining dense and sparse retrieval methods:
   [Hybrid Search in Pinecone](https://docs.pinecone.io/docs/hybrid-search)

6. **LangChain Documentation**:  
   Explore LangChain for building applications with LLMs, handling chains, agents, and retrieval:
   [LangChain Docs](https://docs.langchain.com/)

7. **Hands-On Large Language Models (Book)**:  
   A great resource for understanding and working with large language models:
   [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098122233/) see ch8


