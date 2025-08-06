import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# --- Load Vector DB (created from PDFs) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# --- Load Free Local Model ---
qa_model = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512, temperature=0)
llm = HuggingFacePipeline(pipeline=qa_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.title("HackRx 6.0 LLM Document Query System")
query = st.text_input("Ask a question about the policy documents")
if query:
    answer = qa_chain.invoke({"query": query})
    st.write("### Answer:")
    st.write(answer["result"])
