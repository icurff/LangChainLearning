import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Vietnamese", "text": "Will you marry me, T ?"})

print("Model is typing: ", end="", flush=True)
for chunk in model.stream(prompt):
    print(chunk.content, end="", flush=True)
