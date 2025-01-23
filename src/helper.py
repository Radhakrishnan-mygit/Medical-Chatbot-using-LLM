from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
   loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
   documents=loader.load()
   return documents

def text_split(extracted_data):
   splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
   chunks=splitter.split_documents(extracted_data)
   return chunks

def download_embedding():
   embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
   return embedding