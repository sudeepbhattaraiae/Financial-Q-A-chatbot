import os
import re
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import create_engine, text
from llama_parse import LlamaParse
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Retrieve necessary variables
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Set up database connection
connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)

def table_to_text(df):
    column_names = df.columns.tolist()
    rows = df.values.tolist()
    formatted_text = ""
    for row in rows:
        for head, cell in zip(column_names[1:], row[1:]):
            if cell and not pd.isna(cell):
                formatted_text += f"The {column_names[0]}:{row[0]} of {head} is {cell}. "
    return formatted_text

def extract_table_from_db(table_name, engine):
    with engine.connect() as conn:
        conn.execute(text("USE arithmetic_questions"))
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df

def prepare_context(engine):
    context = ""
    with engine.connect() as conn:
        conn.execute(text("USE arithmetic_questions"))
        tables = conn.execute(text("SHOW TABLES")).fetchall()
        table_names = [table[0] for table in tables]
        for table_name in table_names:
            df = extract_table_from_db(table_name, engine)
            df.name = table_name
            context += table_to_text(df) + "\n\n"
    return context

def generate_answer_from_table_context(question, context):
    prompt = f"""
    Given the following context:
    {context}
    Please answer the following question:
    {question}
    Provide a detailed answer based on the information in the context. If the answer cannot be fully determined from the given information, explain what is known and what additional information might be needed.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {"role": "system", "content": "You are a mathematics assistant that answers questions based on context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def generate_answer_from_text_context(question, context):
    prompt = f"""
    Given the following context:
    {context}
    Please answer the following question:
    {question}
    Provide a short and accurate answer as possible from context based on question.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {"role": "system", "content": "You are a answer finder assistant that answers questions based on context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def get_similar_sentences(question, context, top_n=5):
    sentences = re.split(r'\.\s+', context)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question] + sentences)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [(sentences[i], cosine_similarities[i]) for i in top_indices]