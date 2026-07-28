import ollama
import streamlit as stl
import os

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

model = "gemma3:27b"
embed = "embeddinggemma"
store_name = "ifixitrag"
directory = "./ifixit"
persist="./chromaifixit"

def ingest(file_path):
    if os.path.exists(file_path):
        loader = DirectoryLoader(file_path, glob='./*.pdf',loader_cls=UnstructuredPDFLoader)
        data = loader.load()
        return data
    else:
        stl.error("File not found")

def split(files):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
    chunks = splitter.split_documents(files)
    return chunks

@stl.cache_resource
def load_vector():
    ollama.pull(embed)
    embedding = OllamaEmbeddings(model=embed)
    if os.path.exists(persist):
        vector_db = Chroma.from_documents(
        collection_name=store_name,
        embedding_function = embedding
        )
    else:
        data = ingest(directory)
        if data is None:
            return None
        
        chunks = split(data)

        vector_db = Chroma.from_documents(
        documents=chunks,
        embedding = embedding,
        collection_name = store_name
        )

    return vector_db

def retrieve(vector_db,model):
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI electronics repair assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant information from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines. Be specific
    and thorough in your responses, including any small or optional parts of the response.
    Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), model, prompt = QUERY_PROMPT
    )

    return retriever

def build_chain(retriever,model):
    ragtemplate = """Answer the question based only on the following context, prioritising information from the context, and only using other sources if none is available from the context: {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(ragtemplate)

    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )

    return chain

def main():
    stl.title("Repair Guides")

    user_input = stl.text_input("What would you like to know?", "")

    if user_input:
        with stl.spinner("Let me check that for you..."):
            try:
                llm = ChatOllama(model=model)
                vector_db = load_vector()
                if vector_db is None:
                    stl.error("Unable to load the vector database. Please try again.")
                    return
                retriever = retrieve(vector_db,llm)
                chain = build_chain(retriever,llm)
                response = chain.invoke(input=user_input)
                stl.markdown("**Assistant:**")
                stl.write(response)

            except Exception as e:
                stl.error(f"Sorry, something went wrong : {str(e)}")
    else:
        stl.info(f"Please ask a question to begin")

if __name__ == "__main__":
    main()