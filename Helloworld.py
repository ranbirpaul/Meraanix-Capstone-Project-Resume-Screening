# Create a simple streamlit prompt to allow user to ask any question
import streamlit as st
import os
import patoolib as pl
import chardet

os.environ["OPENAI_API_KEY"] = "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"
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

#!unzip "C:\meraanix project\Resumes.zip"
#pl.extract_archive("C:\meraanix project\Resumes.zip",outdir="C:\meraanix project/resumes")

text_loader_kwargs={'autodetect_encoding': True}
#loader = DirectoryLoader("D:/Parser API Docments/Machine & Deep Learning Notes/ML Fundamentals/Meraanix Capstone Project Resume Screening/Resumes/Text Output/",glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs) 

resumes = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(resumes)

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

# Create a session (Memory based) to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat data from session
# Get each message from messages array
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
   
# Create a session (DB based) to store chat history
prompt = st.chat_input("Ask something")
if prompt:
    with st.chat_message("user"):
      st.write(prompt)

    # Add the user prompt to the chat history session
    st.session_state.messages.append({"role":"user","content": prompt})
      
    llm_response = qa_chain(prompt)
    st.write(llm_response["result"]) 
    st.session_state.messages.append({"role":"bot","content": llm_response["result"]})
    
    # use llm.predict to get the answer
    #answer = turbo_llm.predict(question).strip()
    #st.write(answer)
    
   # with st.chat_message("Manager", avatar=''):
    #  st.write(prompt)
    #  Test  