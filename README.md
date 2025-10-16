# Gen-AI-AskURL
An end-to-end RAG pipeline built with Streamlit that scrapes text from any URL, chunks it, generates embeddings with Hugging Face, stores them in FAISS, and retrieves contextually relevant answers.


A Streamlit app that uses Retrieval-Augmented Generation (RAG) to analyze and answer questions about online news articles using Gemini (Google Generative AI) and FAISS vector search.

🚀 Features

Fetches and extracts text from up to 3 article URLs

Splits content into semantic chunks

Generates embeddings using Hugging Face models

Stores them locally in a FAISS vector database

Answers questions using Gemini 2.5 Flash

Displays sources for transparency

🧠 Tech Stack

Streamlit – Frontend UI

LangChain – RAG framework

FAISS – Vector similarity search

Hugging Face / Gemini – Embeddings & LLM

dotenv – API key management

⚙️ Setup Instructions

Clone repo
1) git clone https://github.com/yourusername/news-research-tool.git
cd news-research-tool

2) Install dependencies

3) pip install -r requirements.txt


4) Add environment variables
->Create a .env file:

GEMINI_API_KEY=your_google_genai_api_key


Run the app

streamlit run app.py

🧩 How It Works

Enter up to 3 article URLs in the sidebar.

Click Process URLs → builds FAISS index (saved as faiss_store.pkl).

Ask a question in the main section → retrieves top chunks → Gemini generates an answer.

Displays summarized answer + sources.

📁 Project Structure
.
├── app.py               # Main Streamlit app
├── faiss_store.pkl      # Saved FAISS vectorstore (auto-created)
├── requirements.txt
├── .env
└── README.md

🔑 Notes

Requires a valid Gemini API key (Google Generative AI).

You can switch to GoogleGenerativeAIEmbeddings instead of Hugging Face if desired.

Use FAISS.save_local() for safer persistence instead of pickle.

🧑‍💻 Author

Gaurav Sharma
AI Engineer | LLM Developer