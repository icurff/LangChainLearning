from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt = ChatPromptTemplate.from_template(
    "Write a concise summary of the following:\n\n{context}"
)

chain = create_stuff_documents_chain(llm, prompt)
# result = chain.invoke({"context": docs})
# print(result)

for token in chain.stream({"context": docs}):
    print(token, end="|")