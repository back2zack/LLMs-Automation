import os 
import logging
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO)
DOC_PATH = "./data/Merged_Documents.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(pdf_path):
    """load my pdf""" 
    if os.path.exists(pdf_path):
        loader = UnstructuredPDFLoader(file_path=pdf_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data 
    
    else:
        logging.error(f"File not found or not of type PDf at path : {pdf_path}")
        return None
        

def split_documents(document):
    """split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(document)
    logging.info(" Document was splited successfully.")
    return chunks

def Generate_vector_db(chunks):
    """Creat a vector database from document chunks."""
    # first pull the embedding model 
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("vector database iitiated successfully")
    return vector_db


def Generate_Retriever(vector_db, llm):
    """Create a multi-query retriever"""
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriver is created successfully")
    return retriever

def Generate_Chain(retriever, llm):

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created successfully.")
    return chain


def main():
    # load document
    doc = ingest_pdf(DOC_PATH)

    # split to chunks 
    chunks = split_documents(document=doc)

    # generate vecrot_db
    vector_DB = Generate_vector_db(chunks=chunks)
    llm = ChatOllama(model=MODEL_NAME)

    # get _Retriever
    Res =  Generate_Retriever(vector_db=vector_DB, llm=llm)

    chain = Generate_Chain(retriever=Res, llm=llm)

    res =  chain.invoke(input=("What is the name of the company?"))
    print("Response:")
    print(res)



if __name__ == "__main__":
    main()




