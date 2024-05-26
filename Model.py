from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



def invoke_chain(chain, question):
    print("Invoking the RAG chain...")
    try:
        response = chain.stream(question)
        print("Chain invocation completed.")
        print("="*30)
        return response
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return "Error processing your request."

def setup_ollama_language_model_chain(vectorstore: Chroma, LLM_name: str, topk: int):
    print(">>>chaining model:", LLM_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": topk})
    llm = ChatOllama(model=LLM_name, temperature=0)
    template = """
                You are a Lagacy CRM contract manageer for Redhat, Answer the question based on the following context:
                {context}

                Question: {question}.

                provide the file path of only the relevent files. 
                for example "Relevent Files based on query: 1. Contract1.pdf 2. Contract2.pdf etc """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content + doc.metadata['file_path']  for doc in docs)
        

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print(">>>Chain setup completed.")
    print("="*30)
    return rag_chain