import sys
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from openai import AzureOpenAI
import configparser
from data import wikipedia_article_on_curling, form1095
import pandas as pd
import re
import numpy as np

# models
DEPLOYMENT_NAME = "azure-oai-test-deployment"
EMBEDDINGS_MODEL = "azure-oai-embeddings"

config = configparser.ConfigParser()
config.read('config.ini')
# ==================== Creating Azure OpenAI Client ==============================
AOAIClient = AzureOpenAI(
    api_key=config['API']["AZURE_OPENAI_KEY"],
    api_version="2023-05-15",
    azure_endpoint=config['API']["AZURE_API_ENDPOINT"]
)

def get_openai_client_response(model, system_prompt, query):
    response = AOAIClient.chat.completions.create(
    model=model, 
    max_tokens=1024,
    messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]
    )
    return response.choices[0].message.content

# query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
# The above query doesnt return any answer as it doesn't have context. So we need to add context to the query 

# Adding the context data about olympic curling to the prompt.
# Eg 1:
system_prompt = "You answer questions about the 2022 Winter Olympics."
query = f"""Use the below article on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article:
\"\"\"
{wikipedia_article_on_curling}
\"\"\"

Question: Which athletes won the gold medal in curling at the 2022 Winter Olympics?"""

#response = get_openai_client_response(DEPLOYMENT_NAME, system_prompt, query)
#print(response)

# Eg 2:
system_prompt = "You answer questions about the 1095 form. Only answer the questions from the text provided and don't add any additional information."
query = f"""Use the below text to answer the subsequent question. If the answer cannot be found, write "I don't know."
\"\"\"
{form1095}
\"\"\"
Question: What are the different sections in the form?"""

#response = get_openai_client_response(DEPLOYMENT_NAME, system_prompt, query)
#print(response)


# ============================= Tokenizing Data ==============================================
# https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/embeddings?tabs=python-new%2Ccommand-line&pivots=programming-language-python

df = pd.read_csv('samples/bill_sum_data.csv')
df_bills = df[['text', 'summary', 'title']]

pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]

# Generate embeddings using Azure OpenAI
def generate_embeddings(text, model=EMBEDDINGS_MODEL): # model = "deployment_name"
    return AOAIClient.embeddings.create(input = text, model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / np.multiply(np.linalg.norm(a), np.linalg.norm(b))

df_bills['ada_v2'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = EMBEDDINGS_MODEL))

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = generate_embeddings(
        user_query,
        model=EMBEDDINGS_MODEL # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )

    return res

if __name__ == "__main__":
    query = 'Can I get information on cable company tax revenue?'
    res = search_docs(df_bills, query, top_n=4)
    print(res["summary"][9]) 