import openai
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

print(openai.__version__)

endpoint = config['API']["AZURE_API_ENDPOINT"]
api_key = config['API']["AZURE_OPENAI_KEY"]
deployment =  "azure-oai-test-deployment"

client = openai.AzureOpenAI(
    base_url=f"{endpoint}/openai/deployments/{deployment}/extensions",
    api_key=api_key,
    api_version="2023-08-01-preview",
)

def answer(query):
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": query,
            },
        ],
        extra_body={
            "dataSources": [
                {
                    "type": "AzureCognitiveSearch",
                    "parameters": {
                        "endpoint": config["API"]["SEARCH_ENDPOINT"],
                        "key": config["API"]["SEARCH_KEY"],
                        "indexName": "azureaiindex",
                    }
                }
            ]
        }
    )
    return completion.choices[0].message.content

while True:
    user_input = input("Enter a query: ")
    if user_input == "exit":
        break

    query = user_input
    try:
        res = answer(query)
        print(res)
    except Exception as err:
        print('Exception occurred. Please try again', str(err))