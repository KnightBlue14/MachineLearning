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


doc_path = "frankenstein.pdf"
model = "llama3.1"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("document loaded...")
else:
    print('load a file')




split = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = split.split_documents(data)

#print(f"Number of chunks: {len(chunks)}")
print("document split...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text"),
    collection_name = "book"
)

print("database built...")



llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant. Your task is to help your user understand a book. They will need help understanding the plot, characters, themes, and context of the book, which you will find in a vector database. Original question: {question}"""
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

prompt1 = "Summarise the events of the book"
prompt2 = "What is the name of Victor Frankenstein's cousin?"
prompt3 = "What are the major themes of the book?"

for prompt in [prompt1,prompt2,prompt3]:
    res = chain.invoke(input = (f"{prompt}"))
    with open("readbook.txt", "a") as f:
        f.write(f"{prompt} \n \n")
        f.write(f"{res} \n \n")

print("%s seconds" % (time.time() - start))
