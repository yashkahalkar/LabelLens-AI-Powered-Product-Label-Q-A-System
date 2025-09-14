import streamlit as st
import cv2
import numpy as np
import easyocr
from pinecone import Pinecone, ServerlessSpec  ## <<< CHANGE: Correct import for native Pinecone client
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore  ## <<< CHANGE: Correct import for LangChain adapter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import asyncio

# --- 1. SETUP AND INITIALIZATION ---

@st.cache_resource
def init_easyocr():
    # Adding Hindi ('hi') along with English ('en')
    return easyocr.Reader(['en'], gpu=False)

reader = init_easyocr()

# --- Initialize LangChain and Pinecone ---
try:
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    # Set up Google AI models
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Note: Changed from "2.5-flash" to "1.5-flash" as it's the current valid model name
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )

    # Set up Pinecone
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    index_name = "product-labels"

    ## <<< CHANGE: Correct Pinecone initialization and index creation logic
    # 1. Initialize the native Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # 2. Check if the index exists and create it if it doesn't
    if index_name not in pc.list_indexes().names():
        st.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension for text-embedding-004
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # 3. Instantiate the LangChain PineconeVectorStore object
    # This object is used to interface with the index for RAG
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)

except Exception as e:
    st.error(f"Error initializing services: {e}. Please check your API keys and configurations in secrets.toml.")
    st.stop()


# --- 2. IMAGE PREPROCESSING AND OCR FUNCTION ---

def preprocess_and_extract_text(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.medianBlur(gray_image, 3)
        binary_image = cv2.adaptiveThreshold(
            denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        result = reader.readtext(binary_image, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None

# --- 3. RAG CHAIN SETUP ---

def setup_rag_chain(vector_store_retriever):
    prompt_template = """
    You are an expert assistant for analyzing product labels.
    Your task is to answer questions based ONLY on the provided text from a product label.
    If the information is not in the text, you must state "Information not found on the label."
    Do not make up information or use external knowledge.

    CONTEXT FROM LABEL:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    rag_chain = (
        {"context": vector_store_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Product Label Q&A", layout="wide")
st.title("📦 Product Label Analyzer")
st.write("Upload an image of a product label, and I'll answer your questions about it!")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a product label image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Label", use_column_width=True)

        if st.button("Process Image"):
            with st.spinner("Processing image and building knowledge base..."):
                image_bytes = uploaded_file.getvalue()
                extracted_text = preprocess_and_extract_text(image_bytes)
                st.session_state.extracted_text = extracted_text

                if extracted_text:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    text_chunks = text_splitter.split_text(extracted_text)

                    ## <<< CHANGE: Correctly add texts using the vector_store object
                    # This now correctly uses the PineconeVectorStore instance to handle embedding and upserting.
                    vector_store.add_texts(text_chunks)
                    st.success("Image processed! The knowledge base is ready.")

                    # Store the retriever in the session state
                    st.session_state.retriever = vector_store.as_retriever()
                else:
                    st.error("No text could be extracted from the image. Please try a clearer image.")

if "extracted_text" in st.session_state and st.session_state.extracted_text:
    with col1:
        with st.expander("View Extracted Text"):
            st.write(st.session_state.extracted_text)

with col2:
    st.header("2. Ask a Question")
    if "retriever" in st.session_state:
        question = st.text_input("e.g., How many calories per serving?", key="question_input")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Searching for the answer..."):
                    rag_chain = setup_rag_chain(st.session_state.retriever)
                    answer = rag_chain.invoke(question)
                    st.subheader("Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload and process an image first.")