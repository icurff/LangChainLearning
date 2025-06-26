import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
    )
    language: str = Field(description="The language the text is written in")

# Structured LLM
structured_llm = model.with_structured_output(Classification)

inp = "ae minh cu rua thau hehehe"
prompt = tagging_prompt.format(input=inp)
response = structured_llm.invoke(prompt)
print(response.model_dump())