import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

 # Load pdf file
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# Split the pdf file
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

#  Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001")


#  Chrome vectordb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally
)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

#
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 1},
# )
#
# results = retriever.batch(
#     [
#         "How many distribution centers does Nike have in the US?",
#         "When was Nike incorporated?",
#     ],
# )
#
# print(results)