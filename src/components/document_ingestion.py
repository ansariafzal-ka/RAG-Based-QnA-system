from langchain_openai import OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv('MODEL')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL')

class DocumentIngestion:
    def __init__(self, pdf_path):
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.pdf_path = pdf_path
        self.loader = PyMuPDF4LLMLoader(file_path=self.pdf_path)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )

    def load_document(self):
        docs = self.loader.load()
        return docs

    def initialise_chunking(self):
        docs = self.load_document()
        chunks = self.splitter.split_documents(docs)
        return chunks
    
    def generate_embeddings(self):
        if os.path.exists('vectorstore'):
            shutil.rmtree('vectorstore')

        chunks = self.initialise_chunking()
        vector_store = Chroma.from_documents(
            chunks,
            self.embedding_model,
            persist_directory='vectorstore'
        )
        print('Embedding stored.')
        return vector_store
    
if __name__=='__main__':
    # pdf_path = 'database/Ansari_Mohammed_Afzal_CS4700_Dissertation_Report.pdf'
    pdf_path = 'database/Ansari Mohammed Afzal_Resume.pdf'
    print('Loading Document...')
    document_ingestor = DocumentIngestion(pdf_path)
    print('Creating Chunks...')
    chunks = document_ingestor.initialise_chunking()
    print('Generating Embeddings...')
    vector_store = document_ingestor.generate_embeddings()