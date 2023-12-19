import os
import logging
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GooglePalm
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# https://data.world/opensnippets/ebay-uk-products-dataset
products = pd.read_csv('https://query.data.world/s/q533sypivreoqflj4asecuqpytcif4?dws=00000', usecols=['name'])


def get_user_query():
    raw_query = input("Enter your query: ").strip()
    if not raw_query:
        logging.error("Please provide a valid query.")
        return None
    return raw_query


def run_semantic_search(user_query):
    # llmChain prompt template for generating a list of 10 items
    prompt_template = '''<s>[INST] <<SYS>>
    Only tell me the best suited product names. The answer should only include ten names.
    <</SYS>>

    {context}[/INST]'''

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    product_names = products['name'].values.astype(str)
    product_embeddings = FAISS.from_texts(product_names, embeddings)

    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if google_api_key is None:
        logging.error("Google API key not found in environment variables.")
        return

    palm_llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
    llm_chain = LLMChain(prompt=PROMPT, llm=palm_llm)

    # Run Semantic Search using llmChain
    enhanced_query = llm_chain.run({'context': user_query})
    logging.info(f"Enhanced Query: {enhanced_query}")

    results = product_embeddings.similarity_search_with_score(enhanced_query, k=10)
    return results


if __name__ == "__main__":
    user_query = get_user_query()
    if user_query:
        res = run_semantic_search(user_query)
        print(res)