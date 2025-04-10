from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader


doc_path = "./data/Merged_Documents.pdf"
model = "llama3.2"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading....")
else:
    print("Upload a PDF file")

    # Preview first page
content = data[0].page_content
print(content[:100])
print("DONE:---------------------------")

#  extract text from file and splot into small chunks 

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting")


import ollama


ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents= chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done saving vector database")

# Retrival

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use 
llm = ChatOllama(model=model)


# here i use the PromptTemplate to help me generate multiple questions from a single question
# this will help find the best match of the needed documents 

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


# MultiQueryRetriver is responsible for first generating multi variants of the given question using the Query_propmt
# and then use these question to extract or find the relevent infos in the files (maybe using cosine similarity or classic euclidian distance)
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)


# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain =  (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

res = chain.invoke(input=("Welche Aufgaben hat Herr Barhdadi in seinem ersten tätigkeit bei dspace?",))

print(res)
print('Done !')