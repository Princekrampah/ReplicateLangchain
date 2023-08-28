from langchain.llms import Replicate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from decouple import config
import os

REPLICATE_API_TOKEN: str = config("REPLICATE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

prompt: str = "Tell me a short story about Issac Newton"

_ = llm(prompt=prompt)
