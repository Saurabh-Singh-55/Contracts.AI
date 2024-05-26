from Embed import *
from Extract import *
from Model import *
from Retriever import *
import streamlit as st
import ollama
from dotenv import load_dotenv
load_dotenv()



with st.sidebar:
    st.title('Document Query Interface')
    # Check if the API key is valid
        
    list_response = ollama.list()
    # Extract the 'name' field from each model's details
    pulled_models = [model['name'] for model in list_response['models']]
    selected_model = st.sidebar.selectbox('Select a Model', list(reversed(pulled_models)))
    st.session_state.topk = st.number_input("Top K Retrival", min_value=1, max_value=10, value=3, help='Number of chunks Retrieved')


    st.markdown('---')
    source_directory = st.text_input("Directory path containing PDFs:")
    if st.button("Load Documents"):
        if not os.path.exists(source_directory):
            st.error("Invalid directory path!")
        else:
            st.session_state.total_pages = extract_pages(source_directory)
            st.success("PDF Pages loaded successfully.")
    st.markdown('---')
    if st.button("Create Embeddings"):
            st.session_state.vectorstore = store_embeddings_local(st.session_state.total_pages)
            st.success("Created Embeddings stored.")

    st.markdown('---')
    if st.button("Load Embeddings"):
        if 'vectorstore' not in st.session_state: 
                st.session_state.vectorstore = load_vectorstore("Vec_Store")
        st.success('Pre-stored VectorStore is Loaded')
            
    


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore("Vec_Store")

# Initialize the RAG chain if it doesn't exist
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_ollama_language_model_chain(st.session_state.vectorstore, selected_model,topk=st.session_state.topk)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Main chat interface logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capture the user's input

if prompt := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
            docs = retriever.get_relevant_documents(prompt)
            with st.expander("See Context"):
                for doc in docs:
                    st.write(doc.page_content)
                    file_path = doc.metadata.get('file_path', None)
                    if file_path:
                        st.markdown(f"**File Path:** `{file_path}`")
                    else: 
                        st.markdown(f"**File Path:** `{'Raptor Cluster Summary File'}`")
            response = invoke_chain(st.session_state.rag_chain, prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
        
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)