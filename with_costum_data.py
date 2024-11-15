# # with pdf terminal based ------------------------------------------------------------------------------------------------     

# import os
# import time
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from dotenv import load_dotenv
# from gtts import gTTS
# import tempfile
# import wave
# import numpy as np
# import sounddevice as sd
# from pydub import AudioSegment

# load_dotenv()

# MODEL_ID = os.environ['MODEL_ID']

# # Load the PDF and vector store setup
# if "vector" not in globals():

#     # Initialize Ollama embeddings with model name
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     # Load the local PDF file instead of a URL
#     loader = PyPDFLoader("/Users/macbook/Desktop/texttoaudio/data/paper1.pdf")
#     docs = loader.load()

#     # Split documents into chunks for efficient searching
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents(docs)

#     # Create FAISS vector store from the documents and embeddings
#     vector = FAISS.from_documents(documents, embeddings)

# # Initialize the Groq-based LLM
# llm = ChatGroq(
#     groq_api_key=MODEL_ID, 
#     model_name='llama3-70b-8192'
# )

# # Define the prompt template
# prompt_template = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context. 
# Think step by step before providing a brief, concise, one-paragraph response to all user queries. 
# I will tip you $200 if the user finds the answer helpful. 
# <context>
# {context}
# </context>

# Question: {input}""")

# # Create a chain for processing the documents and query
# document_chain = create_stuff_documents_chain(llm, prompt_template)

# # Set up the retriever and retrieval chain
# retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# def play_fast_audio(answer, speed_factor=1.2, lang='en'):
#     """Convert the answer to speech and play it as audio"""
#     # Convert the answer to speech and save as MP3
#     tts = gTTS(text=answer, lang=lang, slow=False)
#     mp3_file = "generated_answer.mp3"
#     tts.save(mp3_file)

#     # Load the MP3 file and speed it up using pydub
#     audio = AudioSegment.from_mp3(mp3_file)
#     speedup_audio = audio.speedup(playback_speed=speed_factor)

#     # Save the fast audio as WAV (required for sounddevice playback)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
#         temp_wav_file_name = temp_wav_file.name
#         speedup_audio.export(temp_wav_file_name, format="wav")

#         # Read the WAV file into a numpy array
#         with wave.open(temp_wav_file_name, 'rb') as f:
#             # Get audio file properties
#             framerate = f.getframerate()
#             num_frames = f.getnframes()

#             # Read audio frames into a numpy array
#             audio_data = np.frombuffer(f.readframes(num_frames), dtype=np.int16)

#         # Play the adjusted audio in real-time with sounddevice
#         sd.play(audio_data, samplerate=framerate)
#         sd.wait()  # Wait until the audio finishes playing

#     # Optionally, remove the temporary MP3 and WAV files
#     os.remove(mp3_file)
#     os.remove(temp_wav_file_name)

# # Main logic to interact with the user
# def main():
#     print("Chat with Documents :)")
    
#     # Allow user input for questions
#     user_query = input("Input your prompt here: ")
#     if user_query:
#         print("Processing your query...")

#         # Time the response for performance
#         start = time.process_time()
#         response = retrieval_chain.invoke({"input": user_query})
#         print(f"Response time: {time.process_time() - start} seconds")
        
#         # Extract and print the answer
#         answer = response["answer"]
#         print(f"Generated Answer: {answer}")

#         # Play the answer as audio
#         play_fast_audio(answer)

#         # Display the relevant document chunks (Optional)
#         print("\nDocument Similarity Search:")
#         for i, doc in enumerate(response["context"]):
#             print(f"Document {i+1}:")
#             print(doc.page_content)
#             print("--------------------------------")

# if __name__ == "__main__":
#     main()


# with pdf streamlit based ------------------------------------------------------------------------------------------------     


# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader  # Changed to PyPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# import time
# from dotenv import load_dotenv

# load_dotenv()  #

# MODEL_ID = os.environ['MODEL_ID']

# # Change the document loading logic to load from local PDF
# if "vector" not in st.session_state:

#     # Initialize Ollama embeddings with model name
#     st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     # Load your local PDF file instead of a URL
#     st.session_state.loader = PyPDFLoader("/Users/macbook/Desktop/texttoaudio/data/paper1.pdf")
#     st.session_state.docs = st.session_state.loader.load()

#     # Split documents into chunks for efficient searching
#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

#     # Create FAISS vector store from the documents and embeddings
#     st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

# # Set up the app interface
# st.title("Chat with Docs :) ")

# # Initialize the Groq-based LLM
# llm = ChatGroq(
#     groq_api_key=MODEL_ID, 
#     model_name='llama3-70b-8192'
# )

# # Define the prompt template
# prompt = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context. 
# Think step by step before Provide a brief, concise, one-paragraph responses to all user queries. 
# I will tip you $200 if the user finds the answer helpful. 
# <context>
# {context}
# </context>

# Question: {input}""")

# # Create a chain for processing the documents and query
# document_chain = create_stuff_documents_chain(llm, prompt)

# # Set up the retriever and retrieval chain
# retriever = st.session_state.vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Allow user input for questions
# prompt = st.text_input("Input your prompt here")
# print(prompt)

# # Process the prompt if entered
# if prompt:
#     # Time the response for performance
#     start = time.process_time()
#     response = retrieval_chain.invoke({"input": prompt})
#     print(f"Response time: {time.process_time() - start}")
#     print(response["answer"])

#     # Display the response from the model
#     st.write(response["answer"])

#     # With a streamlit expander, show the relevant document chunks
#     with st.expander("Document Similarity Search"):
#         # Display the documents that were retrieved
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")


# with audio and with indexing ------------------------------------------------------------------------------------------------     


import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import wave
import numpy as np
from pydub import AudioSegment
import requests

load_dotenv()

MODEL_ID = os.environ['MODEL_ID']
PDF_DIRECTORY = "/Users/macbook/Desktop/texttoaudio_deploy/data2"  # Directory with multiple PDFs
INDEX_PATH = "/Users/macbook/Desktop/texttoaudio_deploy/faiss_index"  # Path to save/load FAISS index
OLLAMA_URL = "http://127.0.0.1:11434"  # Connect to local Ollama server

# Initialize embeddings if not in session state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Function to check Ollama server connection
def check_ollama_server(url):
    try:
        # Send a GET request to the base URL (not /status)
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False

# Check Ollama server connection
server_connected = check_ollama_server(OLLAMA_URL)
if server_connected:
    st.write(f"Ollama server connected at {OLLAMA_URL}")
    print(f"Ollama server connected at {OLLAMA_URL}")
else:
    st.write(f"Failed to connect to Ollama server at {OLLAMA_URL}")
    print(f"Failed to connect to Ollama server at {OLLAMA_URL}")

# Initialize documents and vector store in session state
if "vector" not in st.session_state:
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Initializing...")

    # Load or create FAISS index
    if os.path.exists(INDEX_PATH):
        # Load existing FAISS index
        progress_text.text("Loading FAISS index...")
        st.session_state.vector = FAISS.load_local(INDEX_PATH, st.session_state.embeddings, allow_dangerous_deserialization=True)
        progress_bar.progress(100)
        progress_text.text("FAISS index loaded successfully.")
        print("FAISS index loaded successfully.")

        # Display total number of indexed documents
        total_documents = st.session_state.vector.index.ntotal
        st.write(f"Total number of indexed documents: {total_documents}")
        print(f"Total number of indexed documents: {total_documents}")

        # If documents aren't loaded in session, re-fetch from PDFs and split
        if "documents" not in st.session_state:
            all_docs = []
            pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
            for filename in pdf_files:
                file_path = os.path.join(PDF_DIRECTORY, filename)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                all_docs.extend(docs)
            
            # Split documents into chunks and store in session state
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.documents = st.session_state.text_splitter.split_documents(all_docs)

    else:
        # Create FAISS index from scratch
        all_docs = []
        pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
        for i, filename in enumerate(pdf_files):
            file_path = os.path.join(PDF_DIRECTORY, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            all_docs.extend(docs)

            # Update progress
            progress_percentage = int((i + 1) / len(pdf_files) * 100)
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing {filename} ({i + 1}/{len(pdf_files)})...")

        # Split documents and create FAISS index
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(all_docs)
        st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
        st.session_state.vector.save_local(INDEX_PATH)
        progress_bar.progress(100)
        progress_text.text("FAISS index created and saved successfully.")

        # Display total number of indexed documents after creation
        total_documents = st.session_state.vector.index.ntotal
        st.write(f"Total number of indexed documents: {total_documents}")
        print(f"Total number of indexed documents: {total_documents}")

    progress_text.text("Initialization complete.")

# Display all documents in the index
with st.expander("All Indexed Documents"):
    for i, doc in enumerate(st.session_state.documents):
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_number = doc.metadata.get("page", "Unknown")  # Extract the page number from metadata
        
        # Display the document index, filename, and page number
        st.write(f"Document {i + 1}: {filename} | Page Number: {page_number}")
        st.write(doc.page_content)
        st.write("--------------------------------")


# Set up the app interface
st.title("Chat and talk with Docs :) ")

# Initialize the Groq-based LLM
llm = ChatGroq(
    groq_api_key=MODEL_ID, 
    model_name='llama3-70b-8192'
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

# Create a chain for processing the documents and query
document_chain = create_stuff_documents_chain(llm, prompt)

# Set up the retriever and retrieval chain
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Allow user input for questions
prompt_input = st.text_input("Input your prompt here")
print(prompt_input)

# Function to play audio
def play_fast_audio(answer, speed_factor=1.2, lang='en'):
    # Convert the answer to speech and save as MP3
    tts = gTTS(text=answer, lang=lang, slow=False)
    mp3_file = "generated_answer.mp3"
    tts.save(mp3_file)

    # Load the MP3 file and speed it up using pydub
    audio = AudioSegment.from_mp3(mp3_file)
    speedup_audio = audio.speedup(playback_speed=speed_factor)

    # Save the fast audio as WAV (required for sounddevice playback)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        temp_wav_file_name = temp_wav_file.name
        speedup_audio.export(temp_wav_file_name, format="wav")

        # Read the WAV file into a numpy array
        with wave.open(temp_wav_file_name, 'rb') as f:
            # Get audio file properties
            framerate = f.getframerate()
            num_frames = f.getnframes()

            # Read audio frames into a numpy array
            audio_data = np.frombuffer(f.readframes(num_frames), dtype=np.int16)

        # Play the adjusted audio in real-time with sounddevice
        # sd.play(audio_data, samplerate=framerate)
        # sd.wait()  # Wait until the audio finishes playing

    # Optionally, remove the temporary MP3 and WAV files
    import os
    os.remove(mp3_file)
    os.remove(temp_wav_file_name)

# Process the prompt if entered
if prompt_input:
    # Time the response for performance
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt_input})
    print(f"Response time: {time.process_time() - start}")
    answer = response["answer"]
    print(answer)

    # Display the response from the model
    st.write(answer)

    # With a Streamlit expander, show the relevant document chunks along with file names
    with st.expander("Document Similarity Search"):
        # Display document chunks with their index and page number
        for i, doc in enumerate(response["context"]):
            # Find the index of the current document in st.session_state.documents
            document_index = next(
                (idx for idx, d in enumerate(st.session_state.documents) if d.page_content == doc.page_content), 
                "Unknown"
            )
            filename = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_number = doc.metadata.get("page", "Unknown")  # Extract the page number from metadata
            
            # Display the document index, filename, and page number
            st.write(f"Document {document_index + 1}: {filename} | Page Number: {page_number}")
            st.write(doc.page_content)
            st.write("--------------------------------")

    # Optionally play the answer audio
    # play_fast_audio(answer)

