# For sqllite issue in streamlit cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Create a simple streamlit prompt to allow user to ask any question
import streamlit as st
import os
#import patoolib as pl
import chardet
import sqlite3

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Persistent session for storing chat messages 
# Create sqlit connection
conn = sqlite3.connect("data.db")
c = conn.cursor()

# Database functions
# Create a table 
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS chattable(role TEXT, content TEXT)")

# Add record
def add_data(role,content):
    c.execute('INSERT INTO chattable(role,content) VALUES (?,?)',(role,content))
    conn.commit()

# Get all chat data
def get_all_chats_as_dict():
    allchats = []
    c.row_factory = sqlite3.Row
    c.execute("SELECT * FROM chattable")
    data = c.fetchall()
    unpacked = [{k: item[k] for k in item.keys()} for item in data]
    return unpacked

    # for item in data:
    #    print('Role:' + item["role"])
    #    print('Content:' + item["content"])
    #    allchats.append({"role":item["role"],"content": item["content"]})
    
    # return allchats

def get_all_data():
    c.row_factory = sqlite3.Row
    c.execute("SELECT * FROM chattable")
    data = c.fetchall()
    return data


# Init database
create_table()

# Create a session (Memory based) to store chat history
results = get_all_data()
st.write(results)

if "messages" not in st.session_state:
   st.session_state.messages = get_all_chats_as_dict()

# Display chat data from session
# Get each message from messages array
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#!unzip "C:\meraanix project\Resumes.zip"
#pl.extract_archive("C:\meraanix project\Resumes.zip",outdir="C:\meraanix project/resumes")

#text_loader_kwargs={'autodetect_encoding': True}

#loader = DirectoryLoader("D:/Parser API Docments/Machine & Deep Learning Notes/ML Fundamentals/Meraanix Capstone Project Resume Screening/Resumes/Text Output/",glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs) 

#loader = DirectoryLoader("./Resumes/Text Output/",glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs) 

#resumes = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#texts = text_splitter.split_documents(resumes)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

embedding = OpenAIEmbeddings()

persist_directory = "db"
# vectordb = Chroma.from_documents(
#       documents = texts,
#       embedding = embedding,
#       persist_directory = persist_directory
# )

#vectordb._collection.count()

# vectordb.persist()
# vectordb = None

# Vector db with persistent data
vectordb = Chroma(
     persist_directory=persist_directory,
     embedding_function = embedding
)

# Count total chunks
totalcount = vectordb._collection.count()
#totalcount

retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={"k": 30})

# Similarity Search
#retriever.search_type
# Total items to be returned
#retriever.search_kwargs

turbo_llm = OpenAI(temperature=0, model_name="gpt-4-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm = turbo_llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True
)
  
# Create a session (DB based) to store chat history
prompt = st.chat_input("Ask something")
if prompt:
    with st.chat_message("user"):
      st.write(prompt)

    # Add the user prompt to the chat history session
    st.session_state.messages.append({"role":"user","content": prompt})

    # Add chat details into table chat
    #add_data(**{"role":"user","content": prompt})
    add_data(role="user",content=prompt)

    llm_response = qa_chain(prompt)
    st.write(llm_response["result"]) 
    st.session_state.messages.append({"role":"bot","content": llm_response["result"]})
    # Add response chat details into table chat
    #add_data(**{"role":"bot","content": llm_response["result"]})
    
    add_data(role="bot", content=llm_response["result"])
    
    # use llm.predict to get the answer
    #answer = turbo_llm.predict(question).strip()
    #st.write(answer)
    
   # with st.chat_message("Manager", avatar=''):
    #  st.write(prompt)
    #  Test  
