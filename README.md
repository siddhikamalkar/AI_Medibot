# 🩺🤖 MediBot: AI Doctor with RAG, Vision, and Voice
This project is a sophisticated AI-powered medical consultation application built with Python and the Gradio framework. It acts as an AI "doctor" that can answer patient queries using a combination of voice, text, and image input. The application is powered by a Retrieval-Augmented Generation (RAG) system for grounded medical knowledge, multimodal large language models for image analysis, and voice-to-text and text-to-voice capabilities for a seamless conversational experience.

### ✨ Features
Multimodal Input: Patients can describe their symptoms using voice or text, and can also upload images (e.g., a photo of a rash or a medical report).

Voice Interaction: Integrates with an audio-to-text model for transcribing the patient's query and a text-to-speech model for generating an audible response from the AI doctor.

Retrieval-Augmented Generation (RAG): Uses a pre-built knowledge base (from a medical PDF) to provide accurate and grounded medical advice, reducing the risk of hallucinations.

Emergency Alert System: The AI is programmed to identify and flag potential medical emergencies within its responses, prompting the user to seek immediate care.

Session History: Allows for follow-up questions and provides the option to download the full consultation history as a text file.

Chatbot Evaluation: Includes a separate script to evaluate the AI's performance using metrics like ROUGE, BLEU, and BERTScore.

### 📁 File Structure
build_rag_database_from_pdf.py: This is the initial setup script. It takes a medical knowledge PDF file, processes it, creates text embeddings, and saves them to a FAISS index and a pickle file. This index forms the core of the RAG system.

brain_of_the_doctor.py: The core logic for the AI doctor. It handles interactions with the Groq API, processes the user's query along with any image input and the retrieved medical context, and generates the final text response.

voice_of_the_patient.py: Manages the patient's voice input. It uses the speech_recognition library to record audio from the microphone and the Groq Whisper model to transcribe it into text.

voice_of_the_doctor.py: Responsible for the AI doctor's voice output. It uses gTTS (Google Text-to-Speech) to convert the text response into an audio file, which is then played back to the user.

gradio_app.py: The main application file. It orchestrates the entire process, creating the user interface with Gradio and linking the other modules to handle the consultation flow.

chatbot_evaluation.py: A utility script for evaluating the chatbot's performance. It reads a CSV of simulated responses and calculates various NLP metrics to assess the quality of the AI's answers.

### Install dependencies:
This project requires several libraries. You can install them using pip:

pip install -r requirements.txt

Set up API Keys:
Create a .env file in the project's root directory and add your API keys:

GROQ_API_KEY="your_groq_api_key_here"
ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"

Prepare the RAG Database:
Place your medical knowledge PDF (e.g., The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf) in the specified data path within the build_rag_database_from_pdf.py file. Then run the script to build the RAG index:

python build_rag_database_from_pdf.py

### 🚀 Running the Application
Once the setup is complete, you can launch the application:

python gradio_app.py

This will start a local Gradio server, and a link to the web interface will appear in your console. Open this link in your browser to begin your consultation.
