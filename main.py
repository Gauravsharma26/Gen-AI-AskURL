import os
import pickle
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LangChain bits
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ.setdefault("USER_AGENT", "NewsResearchTool/1.0 (contact: you@example.com)")

PICKLE_PATH = Path("faiss_store.pkl")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="AskURL", page_icon="📰", layout="wide")
st.title("📰 News Research Tool")
st.sidebar.title("Article URLs")


#all the urls that user has input
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article1")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com/article2")
url3 = st.sidebar.text_input("URL 3", placeholder="https://example.com/article3")

process_clicked = st.sidebar.button("⚙️ Process URLs")
status = st.empty()

st.markdown("---")


if process_clicked:
    try:
        urls = [u.strip() for u in [url1, url2, url3] if u and u.strip()]
        
        if not urls:
            st.warning("⚠️ Please enter at least one valid URL.")
        else:
            status.info("📥 Fetching & extracting content from URLs…") 

            #extracting data from the urls using WebBaseLoader
            loader = WebBaseLoader(urls) 
            docs = loader.load()
            
            if not docs or all(not doc.page_content.strip() for doc in docs):
                st.error("No text could be extracted from the provided URLs. Try different URLs.")
            else:
                status.info("✂️ Splitting text into manageable chunks…")
                
                #splitting data into chunks using RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = splitter.split_documents(docs)
                
                st.info(f"✅ Created {len(chunks)} text chunks")

                status.info("🧠 Generating embeddings…")

                #Generating the embeddings using huggingface embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDINGS_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                    show_progress=False,
                )

                status.info("🏗️ Building FAISS vectorstore…")
                store = FAISS.from_documents(chunks, embedding=embeddings)

                status.info("💾 Saving FAISS vectorstore…")
                with open(PICKLE_PATH, "wb") as f:
                    pickle.dump(store, f)

                status.success("✅ Index built & saved successfully!")
                st.balloons()

    except Exception as e:
        status.error(f"❌ Error: {str(e)}")



st.markdown("---")
st.subheader("❓ Ask Questions About Your Articles")

question = st.text_input(
    "Enter your question:",
    placeholder="What is the main topic of these articles?",
    disabled=not PICKLE_PATH.exists()
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_clicked = st.button(
        "🔎 Retrieve & Answer",
        disabled=not question or not PICKLE_PATH.exists(),
        use_container_width=True
    )

if not PICKLE_PATH.exists():
    st.warning("📌 Please process URLs first to create an index.")

if ask_clicked and question:
    try:
        status.info("📂 Loading FAISS vectorstore…")
        with open(PICKLE_PATH, "rb") as f:
            store = pickle.load(f)

        status.info("🔍 Retrieving relevant chunks & generating answer…")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0.2,
        )

        retriever = store.as_retriever(search_kwargs={"k": 5})
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )

        result = chain.invoke({"question": question})
        answer = (result.get("answer") or "").strip()
        source_docs = result.get("source_documents", [])

        status.empty()
        
        st.header("🧠 Answer")
        st.write(answer if answer else "_No answer could be generated._")

        st.subheader("📚 Sources")
        if source_docs:
            unique_sources = {}
            for doc in source_docs:
                source_url = doc.metadata.get("source", "Unknown")
                if source_url not in unique_sources:
                    unique_sources[source_url] = doc.page_content[:100]
            
            if unique_sources:
                for i, (url, preview) in enumerate(unique_sources.items(), 1):
                    st.write(f"**{i}. [{url}]({url})**")
                    st.caption(f"Preview: {preview}...")
            else:
                st.write("_No sources identified._")
        else:
            st.write("_No sources identified._")

        st.success("✅ Done!")

    except Exception as e:
        status.error(f"❌ Error: {str(e)}")