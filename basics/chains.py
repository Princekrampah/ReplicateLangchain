from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from decouple import config
import os
import replicate

# downloading the image
from PIL import Image
import requests
from io import BytesIO
import secrets

REPLICATE_API_TOKEN: str = config("REPLICATE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


dolly_llm: Replicate = Replicate(
    model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"
)

text2ImageGen = Replicate(
    model="stability-ai/sdxl:d830ba5dabf8090ec0db6c10fc862c6eb1c929e1a194a5411852d25fd954ac82",
)

first_prompt: PromptTemplate = PromptTemplate(
    input_variables=["company_name", "product"],
    template="The provided that, {company_name} is a company in Nairobi Kenya that makes {product}. Write me a short 30 word advertisement that the company can you to boost it sales in Kenya.",
)

# the output_key of this chain is the input_variable to the next chain
first_chain: LLMChain = LLMChain(
    llm=dolly_llm,
    prompt=first_prompt,
    output_key="advert_text"
)

second_prompt: PromptTemplate = PromptTemplate(
    input_variables=["advert_text"],
    template="Generate a company logo from the given text: {advert_text}",
)

# the output_key of this chain is the output_variables of the overall chain
second_chain: LLMChain = LLMChain(
    llm=text2ImageGen,
    prompt=second_prompt,
    output_key="image_url"
)


# Run the chain specifying only the input variable for the first chain.
overall_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["company_name", "product"],
    output_variables=["image_url", "advert_text"],
    verbose=True
)

# use .run({}) when dealing with a single input
sequential_chain_response = overall_chain({
    "company_name": "Helton Cloths", "product": "Fashion Clothes"})

# print(sequential_chain_response)

advert_text = sequential_chain_response.get("advert_text")
print(advert_text)

# download image
response = requests.get(sequential_chain_response.get("image_url"))
img = Image.open(BytesIO(response.content))

# save image
random_filename = secrets.token_hex(nbytes=20) + ".png"
img.save(fp=f"./genImages/{random_filename}")

print("image saved")

print("Generating audio, please wait...")

audio_response = replicate.run(
    "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
    input={"prompt": advert_text}
)
# print(audio_response)

print("Downloading audio...")
# download audio
response = requests.get(audio_response.get("audio_out"))

# save image
random_filename = secrets.token_hex(nbytes=20) + ".wav"

with open(f"./genAudio/{random_filename}", 'wb') as f:
    f.write(response.content)


print("Done!")
