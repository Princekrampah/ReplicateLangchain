from langchain.llms import Replicate
from decouple import config
import os

REPLICATE_API_TOKEN: str = config("REPLICATE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llm = Replicate(
    model="replicate/codellama-13b:1c914d844307b0588599b8393480a3ba917b660c7e9dfae681542b5325f228db"
)

prompt: str = "Write me a python code to send emails"
prompt: str = "Write me a Golang code to print hello world"

response: str = llm(prompt=prompt)
print(response)
