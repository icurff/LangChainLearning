import os

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)
class Data(BaseModel):
    people: List[Person]


structured_llm = model.with_structured_output(schema=Data)

text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.format(text = text)
response = structured_llm.invoke(prompt)
print(response)
