from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import chromadb
from chromadb.utils.batch_utils import create_batches
import uuid


# Set the path to the file you'd like to load
path = "./global-restaurant-ratings-and-food-analytics-dataset.csv"

df = pd.read_csv(path)
embeddings = OllamaEmbeddings(model="embeddinggemma")

db_location = "./chrome_langchain_db_restaurant"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Restaurant Name"],
            metadata={"city":row["City"], "cuisines": row["Cuisines"], "cost": row["Average Cost for two"],"currency": row["Currency"],"rating": row["Aggregate rating"], "votes": row["Votes"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)


    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]
            
    split_docs_chunked = split_list(documents, 100)

    for split_docs_chunk in split_docs_chunked:
        vectorstore = Chroma.from_documents(
            documents=split_docs_chunk,
            embedding=embeddings,
            persist_directory=db_location,
        )


vector_store = Chroma(
    collection_name="restaurant_records",
    persist_directory=db_location,
    embedding_function=embeddings
)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)