# Medical Chatbot RAG Application

This is a **Medical Chatbot** built using **Retrieval-Augmented Generation (RAG)**, leveraging **Ollama**, **Flask**, **Hugging Face's Text Embeddings**, **Pinecone Vector Store**, and **PyPDF Loader** for PDF parsing. The chatbot answers medical-related queries about diseases, solutions, and more by retrieving relevant information from a provided PDF document.

## Technologies Used

- **Ollama**: AI-powered language model used for generating human-like responses.
- **Flask**: Lightweight web framework for serving the chatbot and managing API requests.
- **Hugging Face**: Text embedding models for converting medical text into vector representations.
- **Pinecone**: Vector database to store and search text embeddings.
- **PyPDF Loader**: A library for extracting text data from PDF documents.

## How it Works

The application works by using a combination of RAG (retrieval-augmented generation) and text embeddings to provide relevant medical answers.

### Key Features:
1. **PDF Document Parsing (PyPDF Loader)**: 
   - The PDF file you provide is parsed for relevant medical content (diseases, symptoms, treatments, etc.).
   - Text from the PDF is processed and converted into embeddings using a Hugging Face model.

2. **Vector Database (Pinecone)**:
   - The text embeddings are stored in **Pinecone**, a vector store.
   - Pinecone allows for efficient similarity search to find relevant information based on user input.

3. **Medical Query Handling**:
   - When a user asks a medical question, the chatbot first converts the question into an embedding.
   - The application queries Pinecone to retrieve the most relevant text from the PDF document.
   - The relevant text is then passed to the Ollama language model, which generates a response.

4. **Response Generation**:
   - The generated response is returned to the user, providing answers about diseases, solutions, or any other medical query.



