from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os



def store_embeddings_local(documents):
    """Creates and stores embeddings from the provided texts using the BAAI/bge-m3 model.
    Args:
        documents (List[str]): The texts to process and create embeddings for.
        chunk_size (int): Size of the text chunk for processing.
        chunk_overlap (int): Overlap size between chunks.
        level (int): Starting level for processing.
        n_levels (int): Number of levels for hierarchical processing.
    Returns:
        Chroma: The created vector store.
    """
    print(">>> Creating and storing embeddings...")

    # Load the BAAI/bge-m3 model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    # model = AutoModel.from_pretrained('BAAI/bge-m3', torch_dtype=torch.float16)
    # model.to('cuda')  # Move model to GPU

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu", }
    encode_kwargs = {"normalize_embeddings": True}
    embedding = HuggingFaceBgeEmbeddings(
                            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                            )
    # Setup vector store path
    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')
    persist_directory =  vectorstore_path
    ## Here is the nmew embeddings being used

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=50,
        keep_separator=False,
    )
    docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=docs,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    print(">>> Raw Embeddings stored.")
    print("=" * 30)
    
    return vectorstore


def load_vectorstore(path: str) -> Chroma:
    print(f">>>Loading vectorstore.")


    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu", }
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceBgeEmbeddings(
                            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                            )

    vectorstore = Chroma(persist_directory=path, embedding_function=embedding_function)
    print(">>>Vectorstore loaded.")
    print("="*30)
    return vectorstore