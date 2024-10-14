import requests
import openai
import configparser
from PIL import Image
import json
import os

config = configparser.ConfigParser()
config.read('config.ini')

print(openai.__version__)

endpoint = config['API']["AZURE_API_ENDPOINT"]
api_key = config['API']["AZURE_OPENAI_KEY"]
api_version="2023-06-01-preview"

client = openai.AzureOpenAI(
    base_url=f"{endpoint}/openai/images/generations:submit?api-version={api_version}",
    api_key=api_key,
    api_version=api_version,
)

result = client.images.generate(
    model="dalle3", # the name of your DALL-E 3 deployment
    prompt="a man with happy face sitting on a chair",
    n=1
)

json_response = json.loads(result.model_dump_json())

# Set the directory for the stored image
image_dir = os.path.join(os.curdir, 'images')

# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# Initialize the image path (note the filetype should be png)
image_path = os.path.join(image_dir, 'generated_image.png')

# Retrieve the generated image
image_url = json_response["data"][0]["url"]  # extract image URL from response
generated_image = requests.get(image_url).content  # download the image
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)

# Display the image in the default image viewer
image = Image.open(image_path)
image.show()