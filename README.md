# RAG-EnhancedQA

## Overview
This project focuses on implementing and evaluating a **Retrieval-Augmented Generation (RAG)** system, which combines generative language models and information retrieval techniques. Given a question, the system retrieves relevant passages and generates an answer based on those passages. The project explores various RAG approaches to improve retrieval and generation performance, including baseline methods, FAISS-based retrieval, and advanced state-of-the-art methods.

## Objectives
- Implement a **baseline model** using direct prompting without retrieval.
- Develop a **frozen RAG system**, retrieving passages using a static embedding model.
- Optimize retrieval with **FAISS indexing** for efficient passage retrieval.
- Improve our RAG system by leveraging the latest state-of-the-art methods.
- Evaluate the effectiveness of different approaches using **BLEU scores**.

## Dataset
The dataset consists of:
- **texts.csv**: A corpus of passages extracted from various topics such as mathematics, physics, chemistry, biology, computer science, music, and psychology.
  - `id`: Unique identifier for each passage.
  - `text`: The passage content.
- **questions_train.csv**, **questions_val.csv**, **questions_test.csv**:
  - `id`: Unique identifier for each question.
  - `question`: The question text.
  - `text_id` (except for test set): List of relevant passage IDs from `texts.csv`.
  - `answer` (except for test set): The correct answer.

### **Description of the Data**
The passages are short English texts covering various subjects. These texts focus on highly specific topics within a domain. For example, there might be 1000 texts related to computer science, 100 discussing memory, and within those, 5 focusing on memory in a specific processor.

The questions are designed based on particular topics covered in specific texts and can be answered using at least one passage. Some questions require information from multiple passages. The dataset is structured to ensure concise answers (< 30 words), often a short phrase or numerical value.

The dataset includes ~13,000 passages, with ~1,700 question-answer pairs for training and ~500 for validation. The test set contains 500 questions where answers must be predicted.

## Methodology
### 1. Baseline Model
- Generates answers using only the question without retrieved passages.
- Establishes a starting point for performance comparison.

### 2. Frozen RAG System
- Retrieves the top-*k* most relevant passages using a **static embedding model**.
- Uses the retrieved passages as context for the generative model.

### 3. FAISS-Based Retrieval
- Indexes passage embeddings with **FAISS** for efficient similarity search.
- Retrieves passages based on **cosine similarity**.

### 4. Advanced RAG Method
In this section, we aimed to improve both the **retrieval** and **generative** components of our RAG system using state-of-the-art methods.

#### **Generative Component**
- We used **[microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)** as our language model for answer generation.
- We first applied **LoRA (Low-Rank Adaptation)** to fine-tune our generative model, making it more capable of answering questions based on contextual documents while maximizing the use of those documents.
- We explored training a **custom embedding model** using a **contrastive learning approach** on positive and negative examples. However, this method did not yield the expected results, leading us to focus on other architectural improvements.

#### **Retrieval Component**
- We implemented two distinct query augmentation techniques in parallel:
  - **HYDE (Hypothetical Document Embeddings)**: We generated hypothetical documents based on queries and performed retrieval on these generated documents.
  - **Query Decomposition**: We split the original query into two or three sub-questions using our generative model and performed retrieval on those sub-questions.
- We observed that these two methods retrieved **complementary documents** in terms of relevance. To leverage their synergy, we **combined** the results from both approaches.
- Finally, we applied **document reranking** using the **cross-encoder ms-marco-MiniLM-L-6-v** to refine the ranking of retrieved passages.

## Example
For instance, given the question:

> What type of bonds are used to form branches in glycogen?

Instead of generating the answer directly, the RAG approach first retrieves relevant passages. One example passage is:

> **Glycogen Structure and Function**: Glycogen is a molecular polymer of glucose used for energy storage. It is composed of linear chains of glucose molecules linked by α-1,4-glycosidic bonds, with branches formed off the chain via α-1,6-glycosidic bonds. The branches provide additional "free ends" for linear chains, allowing for faster glucose release.

Using this passage, the model generates the following answer:

> α-1,6-glycosidic bonds

## Output
The final output of our model is a CSV file containing two columns:
- **id**: The question identifier from the test dataset.
- **answer**: The generated answer for the corresponding question.

## Evaluation Methods
To evaluate the model's performance, we employed several metrics:
- **BLEU-1**: Measures unigram similarity between generated and reference answers.
- **BLEU-2**: Considers both unigram and bigram overlaps for a more detailed assessment.
- **Precision@k and Recall@k**: Evaluates the effectiveness of passage retrieval.
- **Mean Reciprocal Rank (MRR)**: Assesses the ranking quality of retrieved documents.

## Results and Possible Improvements

### Advantages
- The generative model prompt significantly improved performance in initial experiments.
- Retrieval enrichment introduced diversity compared to the baseline.
- Fine-tuning provided improvement.

### Limitations
- The new retrieval model did not significantly outperform the FAISS-based retrieval.
- Combining retrieval enrichment techniques effectively remains a challenge.
- A non-fine-tuned generative model used as a judge performed poorly.

### Potential Improvements
- **Self-RAG**: Could help decide when retrieval is necessary.
- **Trained Judge**: A model trained on the dataset could assess response sufficiency, enabling iterative retrieval-generation cycles.
- **End-to-End Training**: Aligning retrieval and embeddings could improve overall performance.

## References and Related Work
1. **[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)** → Summarizes advanced and modular RAG methods.
2. **[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)** → Introduces HYDE.
3. **[ARAGOG: Advanced RAG Output Grading](https://arxiv.org/abs/2404.01037)** → Presents the combination of HYDE + LLM reranking.
4. **[Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy](https://arxiv.org/abs/2305.15294)** → Discusses an iterative method for retrieval enhancement.

## Contact
For inquiries or further information, please contact me at mouniralaouiyoussef@hotmail.fr

