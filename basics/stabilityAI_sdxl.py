from langchain.llms import Replicate
from decouple import config
import os

# downloading the image
from PIL import Image
import requests
from io import BytesIO
import secrets

REPLICATE_API_TOKEN: str = config("REPLICATE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

text2ImageGen = Replicate(
    model="stability-ai/sdxl:d830ba5dabf8090ec0db6c10fc862c6eb1c929e1a194a5411852d25fd954ac82",
)

prompt: str = "A picture of a man standing on car"

response_img_url: str = text2ImageGen(prompt=prompt)
print(response_img_url)

# download image
response = requests.get(response_img_url)
img = Image.open(BytesIO(response.content))

# save image
random_filename = secrets.token_hex(nbytes=20) + ".png"
img.save(fp=f"./genImages/{random_filename}")

print("image saved")
