# Author: Jeremy Boyd (jeremy.boyd@va.gov)
# Description: Script that scrapes VA Directive PDFs from the web, extracts all
# pages of text, merges pages into documents, splits documents into chunks,
# embeds chunks, stores chunks in a vector store, writes vector store to disk.

# Packages
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

# Load environment variables from .env file
load_dotenv()

# Access env variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# DB_CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_CeWFPzx08Tac@ep-still-salad-a6zn0w5u-pooler.us-west-2.aws.neon.tech/neondb?sslmode=require")

DB_CONNECTION_STRING="postgresql://neondb_owner:npg_CeWFPzx08Tac@ep-still-salad-a6zn0w5u-pooler.us-west-2.aws.neon.tech/neondb?sslmode=require"


# PDF directory
pdf_dir = "pdf/"

# List for docs
docs = []

# Loop through PDF files in pdf_dir
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        
        # Load PDF & extract text from all pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Merge text from all pages
        merged_text = "\n\n".join([page.page_content for page in pages])
        
        # Convert merged_text to langchain doc with metadata
        merged_doc = Document(page_content = merged_text, metadata = {"source": filename})
        
        # Add to list
        docs.append(merged_doc)

# Check number of PDFs processed
print(f"Processed {len(docs)} PDFs.")

# Preview first 1K chars from first doc
# print(docs[0].page_content[:1000])

# Split docs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
all_splits = text_splitter.split_documents(docs)
print(f"Split docs into {len(all_splits)} sub-documents.")

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Embed docs, put in vector store, write vector store to postgreSQL DB hosted on
# Neon. Looks like if you run this again it adds the entire set of embeddings to the vector store again, so that every doc is in there twice. Need some logic to avoid this.
vector_store = Chroma.from_documents(all_splits, embeddings, persist_directory=DB_CONNECTION_STRING)

# Print number of items in vector store
# This is indicating that I've written vectors into db 3 times
num_items = len(vector_store.get()["ids"])
print(f"Total items in Chroma vector store: {num_items}")
