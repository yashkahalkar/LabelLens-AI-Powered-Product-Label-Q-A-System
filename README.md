# LabelLens-AI-Powered-Product-Label-Q-A-System
An intelligent Q&A assistant that lets you "talk" to your product labels. Simply upload an image of any product package, and ask questions like "How much sugar is in this?" or "Does this contain nuts?" to get instant, accurate answers.
## About The Project
This project addresses the common challenge of quickly extracting specific information from dense and often confusing product labels. LabelLens leverages a modern AI stack, combining Optical Character Recognition (OCR) with a Retrieval-Augmented Generation (RAG) pipeline to provide a seamless, conversational experience for the user.


## Key Features
* 📷 Image-to-Text: Upload any JPG, PNG, or JPEG of a product label for analysis.

* 🧠 Advanced OCR Pipeline: Utilizes a robust OpenCV preprocessing pipeline (grayscale, noise reduction, adaptive thresholding) and EasyOCR to accurately extract text, even from images with varied lighting and angles.

* 💬 Conversational Q&A: Ask questions in natural language and get answers based only on the information present on the label, powered by a Google Gemini RAG system.

* ⚡ Fast & Efficient: Leverages a Pinecone vector database for rapid semantic search, providing answers in seconds.

## How It Works (Architecture)
The application follows an end-to-end pipeline from image input to text output:

* Image Processing: The user uploads an image. It is immediately preprocessed with OpenCV to enhance clarity and standardize it for the OCR engine.

* OCR: EasyOCR scans the processed image and extracts all readable text into a single document.

* Embedding & Storage: The extracted text is split into smaller, meaningful chunks. Google's Gemini API generates vector embeddings for each chunk.

* Vector Indexing: These embeddings are then stored (indexed) in a Pinecone serverless vector database.

* Retrieval & Generation (RAG): When a user asks a question, LangChain orchestrates the RAG pipeline:

  The user's question is converted into a vector embedding.

  A similarity search is performed in Pinecone to retrieve the most relevant text chunks from the label.

  The retrieved chunks (context) and the original question are passed to the Gemini LLM, which generates a precise, context-aware answer.

## Tech Stack
* Frontend: Streamlit

* Backend & Orchestration: LangChain

* LLM & Embeddings: Google Gemini (gemini-1.5-flash, text-embedding-004)

* Vector Database: Pinecone

* OCR Engine: EasyOCR

* Image Processing: OpenCV

## Usage
1. Upload an image of a product label.

2. Click the "Process Image" button and wait for the knowledge base to be built.

3. Once processed, type your question into the input box and click "Get Answer".

## Future Improvements
* Add support for multiple language
* Improve accuracy by using advanced models

##
**Made by Yash Kahalkar with 💻**
