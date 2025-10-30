
import langchain



# LangChain imports
# Loads PDF files into Python so you can process their text.
from langchain_community.document_loaders import PyPDFLoader
# Splits long text into smaller chunks for better processing.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAIEmbeddings It refers to turning words or sentences into numbers so a computer can understand their meaning and compare them efficiently.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# For example, "I love pizza" might be close in vector space to "Pizza is my favorite food"
from langchain.vectorstores import Chroma
# A “do nothing” step in a LangChain pipeline that simply passes input to output.
from langchain_core.runnables import RunnablePassthrough
# Insert variables like {question} into a template so the LLM knows how to respond.
from langchain_core.prompts import ChatPromptTemplate
# BaseModel lets you define structured data models.
# Field allows you to add metadata, validation rules, or descriptions.
from langchain_core.pydantic_v1 import BaseModel, Field

# Standard Python & utility libraries
# : Provides functions to interact with the operating system.
import os
# Creates temporary files/folders that auto-delete — useful for saving uploaded files temporarily.
import tempfile
# Builds interactive web apps for data and AI projects directly from Python.
import streamlit as st
# Handles data analysis and manipulation, especially tables and CSV files
import pandas as pd
# Loads environment variables from a .env file (e.g., API keys for OpenAI).
from dotenv import load_dotenv

# --- process PDF document ---
# load pdf document
loader=PyPDFLoader("data\Design sans titre.pdf")
pages=loader.load()
# split document into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunck_overlap=200,length_function=len,separator={"\n\n","\n"," "})
text_splitter.split_document(pages)  
# text embedding
def get_embedding(text, model_name="nomic-embed-text"):

    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None    
# create vector database
# create new chroma database from the documents
from langchain_community.vectorstores import Chroma
import uuid
def create_and_save_vectorstore(chunks, embedding_function, path="vectorstore"):
    # create a lisz of unique ids for each document based on the content
    ids=[str(uuid.uuidS(uuid.NAMESPACE_DNS,doc.page_content)) for doc in chunks]
    # ensure that only unique ddocs with unique ids are kept 
    unique_ids=set()
    unique_chunks=[]

 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=path
    )

    # Save the database on disk
    vectorstore.persist()

    print(f"✅ Vectorstore saved at '{path}'")
    return vectorstore
vectorstore = create_vectorstore(
    chunks=chunks,
    embedding_function=embedding_function,
    vectorestore_path="vectorstorechroma"
)
# query for relevant data
# load vectorstore
vectorstore = Chroma(
    persist_directory="vector_chroma",
    embedding_function=embedding_function
)
# create retriever and get relevant chunks
# cosinesimilarity how words are related 
retriver =vectorstore.as_retriver(serach_type="similarity")
# ****---change to information related to  cv ----****
relevant_chunks=retriver.invoke("what is the title of article")
# a prompt is the input or instruction you give to the model
PROMPT_TEMPLATE = """
Use the following context to answer the question.
If you don't know the answer, just say you don't know — don't make up anything.

Context:
{context}

Question:
{question}

Answer:
"""
# concatenate context text
context_text="\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])
# create prompt
prompt_template=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text,question="what is the title of the paher?")
print(prompt)
#generate the response
llm.invoke(prompt)
# using langchain expression language
# This function takes a list of documents (from a retriever) and joins their textual content 
# (page_content) into one large string separated by blank lines. 
def format_doc(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# here the input is the cv qnd the output is the list of positions 
# suitqble for your cv
rag_chain(
    {"context":retriver | format_doc,"question":RunnablePassthrough{} }
    | prompt_template
    | llm) 
from pydantic import BaseModel, Field

class ExtractedInfor(BaseModel):
    """Extracted information about a CV"""

    technical_skills: str = Field(description="Technical skills listed in the CV")
    soft_skills: str = Field(description="Soft or interpersonal skills mentioned in the CV")
    education: str = Field(description="Educational background such as degrees, universities, and majors")
    experience: str = Field(description="Work experience details including roles, companies, and durations")
    languages: str = Field(description="Languages known or mentioned in the CV")
    certifications: str = Field(description="Professional certifications or completed courses")
    projects: str = Field(description="Notable academic or professional projects included in the CV")
    contact_info: str = Field(description="Candidate contact details like email, phone number, and address")
rag_chain(
    {"context":retriver | format_doc,"question":RunnablePassthrough{} }
    | prompt_template
    | llm.with_structured_output[ExtractedInfor,strict=True]) 
rag_chain.invoke{"Give me the best postition based on infromation"}
# Transfrom response into a datafrane
