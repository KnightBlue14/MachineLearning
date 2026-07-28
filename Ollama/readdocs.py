import time
import ollama

start = time.time()

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

doc_path = "Greeningpostcovid.pdf"
model = "gemma3:12b"
embed = "embeddinggemma"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("document loaded...")
else:
    print('load a file')



split = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = split.split_documents(data)

print(f"Number of chunks: {len(chunks)}")
print("document split...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding = OllamaEmbeddings(model=embed),
    collection_name = "report"
)

print("database built...")



llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant information from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt = QUERY_PROMPT
)

ragtemplate = """Answer the question based only on the following context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ragtemplate)#

print("prompt generated...")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


prompt1 = "Summarise the report"
prompt2 = "Who are the commitee members mentioned in the report, and which political parties do they belong to?"
prompt3 = "What are the main goals of the report?"

for prompt in [prompt1,prompt2,prompt3]:
    res = chain.invoke(input = (f"{prompt}"))
    with open("readreport.txt", "a") as f:
        f.write(f"{prompt} \n \n")
        f.write(f"{res} \n \n")

print("%s seconds" % (time.time() - start))