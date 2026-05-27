from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.getenv('MODEL')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL')

class RetrievalGeneration:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.retriever = self.vector_db.as_retriever(search_kwargs={'k': 10})
        self.model = init_chat_model(model=MODEL)

    def retrieve_chunks(self, user_query):
        relevant_chunks = self.retriever.invoke(user_query)
        return relevant_chunks
    
    def generate_response(self, user_query):
        relevant_chunks = self.retrieve_chunks(user_query)
        context = '\n\n'.join(
            [chunk.page_content for chunk in relevant_chunks]
        )

        prompt = f'''
            You are a PDF document Q&A assistant.

            Use the provided context from the document to answer the user's question accurately.

            Context:
            {context}

            Question:
            {user_query}
        '''

        response = self.model.invoke(prompt)

        return response.content
    
if __name__ == '__main__':

    embedding_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    vector_db = Chroma(
        persist_directory='vectorstore',
        embedding_function=embedding_model
    )
    print('Generating Response...')
    rag = RetrievalGeneration(vector_db)
    query = 'Summarize this paper, tell me about the title, little but about literature review, methodology, any algorithms mentioned etc.'
    response = rag.generate_response(query)

    print(f'Response:\n {response}')