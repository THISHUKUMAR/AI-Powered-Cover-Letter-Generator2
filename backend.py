import pdfplumber
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your API Key
GEMINI_API_KEY = "AIzaSyB8mDOGyq4kfn5Re2zjygDTqyygxV3uWUg"  # Replace this with your real key

# Load text from PDF
def load_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# Create vectorstore from combined resume and JD text
def create_vectorstore(resume_text, jd_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([resume_text, jd_text])

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore

# Generate formatted cover letter
def generate_cover_letter(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        template="""
You are an expert career assistant.

Write a professional and concise cover letter using the context below.
Structure it as follows:

1. **Opening**: Greet the Hiring Manager and mention the role.
2. **Body**: Highlight relevant skills, experiences, and achievements from the resume aligned with the job description.
3. **Conclusion**: Express enthusiasm and politely close.

Context:
{context}

[Start your cover letter below]
""",
        input_variables=["context"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )

    query = "Generate a personalized and well-structured cover letter."

    chain: Runnable = retriever | (lambda docs: {"context": "\n\n".join(d.page_content for d in docs)}) | prompt | llm

    result = chain.invoke(query)
    
    # Ensure string output
    return result.content if hasattr(result, 'content') else str(result)
