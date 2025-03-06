import os
import faiss
import google.generativeai as genai
import requests
import numpy as np
from bs4 import BeautifulSoup
import nltk

# Giving the Gemini AI API
genai.configure(api_key="AIzaSyCJf4tSbfOcAJAsWXk3Ge3j_4yM0scA1uc")

# Extracting text from a webpage using beautifulsoup
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

# Function to create embeddings using Gemini's embedding model
def get_embedding(text):
    embedding_model_name = 'models/embedding-001'
    response = genai.embed_content(model=embedding_model_name, content=text)
    return response['embedding']

# First I will load web links, then will create embeddings (Using faiss)
def create_knowledge_base(urls):
    texts = []
    chunked_texts = []
    for url in urls:
        page_text = extract_text_from_url(url)
        paragraphs = [p for p in page_text.split('\n') if p.strip()] #extracting relevent paragraphs
        chunked_texts.append(paragraphs)
        texts.extend(paragraphs)

    embeddings = [get_embedding(text) for text in texts]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, texts, chunked_texts

def retrieve_relevant_text(query, index, texts, k=3): 
    query_embedding = np.array(get_embedding(query)).reshape(1, -1) #Getting the relevent texts from those relevent paragraphs
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

def chat_with_knowledge_base(query, index, texts, chunked_texts):
    relevant_texts = retrieve_relevant_text(query, index, texts)

    full_relevant_text = "\n".join(relevant_texts) #combine relevant chunks.(I tried but still not able to analyse the full texts)
    prompt = f"Analyze the following website content and answer the question:\n\n{full_relevant_text}\n\nQuestion: {query}"
    model_name = 'models/gemini-2.0-flash' #Gemini 2.0 Flash Model
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occured: {e}")
        return "Error generating response"

# For Example
urls = ["https://www.holy-bhagavad-gita.org/"]

index, texts, chunked_texts = create_knowledge_base(urls)
query = "Between which two people the conversation is told?" #Querry input by user
response = chat_with_knowledge_base(query, index, texts, chunked_texts)
print(response)
