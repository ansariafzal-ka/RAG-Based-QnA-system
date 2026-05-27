import streamlit as st
import os
from src.components.document_ingestion import DocumentIngestion
from src.components.retrieval_generation import RetrievalGeneration
from datetime import datetime

if 'paper_uploaded' not in st.session_state:
    st.session_state.paper_uploaded = False
    
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = False

if 'retrieval_system' not in st.session_state:
    st.session_state.retrieval_system = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('Document QnA Chatbot')

if not st.session_state.paper_uploaded:

    uploaded_file = st.file_uploader(
        'Upload Document (PDF only)',
        type=['pdf']
    )

    if uploaded_file is not None:
        if st.button('Upload Paper'):
            os.makedirs('database', exist_ok=True)

            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            file_name = f'{timestamp}_{uploaded_file.name}'

            save_path = os.path.join('database', file_name)

            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner('Processing document...'):
                document_ingestor = DocumentIngestion(save_path)
                chunks = document_ingestor.initialise_chunking()
                vector_store = document_ingestor.generate_embeddings()
                st.session_state.retrieval_system = RetrievalGeneration(vector_store)

            st.session_state.paper_uploaded = True
            st.session_state.uploaded_file_path = save_path
            st.session_state.uploaded_file_name = uploaded_file.name

            st.success(f'File uploaded successfully: {uploaded_file.name}')
            st.rerun()
else:
    st.subheader('Ask questions about the uploaded document')
    
    # Display chat history
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        st.chat_message(role).markdown(content)
    
    # Chat input
    query = st.chat_input("Ask anything...")
    
    if query:
        st.session_state.messages.append({'role': 'user', 'content': query})
        st.chat_message('user').markdown(query)
        
        # Generate response using your retrieval system
        response = st.session_state.retrieval_system.generate_response(query)
        
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})