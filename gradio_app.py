import os
import re
import gradio as gr
from dotenv import load_dotenv
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts
from rag_utils import retrieve_context

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# System Prompt
system_prompt = (
    "You are a professional and empathetic AI doctor. "
    "Use the provided medical knowledge to respond accurately. "
    "Be concise, clear, avoid disclaimers. "
    "Recommend possible causes, diagnostic tests, treatment options, and suggest when to seek emergency care. "
    "If an image is provided, incorporate it naturally into your medical advice."
)

# Histories
chat_history = []
conversation_log = []
emergency_keywords = ["chest pain", "difficulty breathing", "severe bleeding", "stroke", "heart attack"]

def check_emergency(response_text):
    return any(keyword in response_text.lower() for keyword in emergency_keywords)

def consult_doctor(audio_filepath, manual_text, image_filepath):
    if not audio_filepath and not manual_text.strip():
        return "Please record or type your symptoms.", "", None

    if manual_text.strip():
        user_query = manual_text.strip()
    else:
        user_query = transcribe_with_groq(audio_filepath)
        if not user_query.strip():
            return "Could not understand audio. Please type your symptoms manually.", "", None

    retrieved_context = retrieve_context(user_query)
    print("🔎 Retrieved Medical Context:\n", retrieved_context)

    full_prompt = f"""
{system_prompt}

Relevant Medical Knowledge:
{retrieved_context}

Patient says: {user_query}

Now, based on the symptoms and knowledge provided, kindly suggest:
- Possible medical conditions
- Recommended diagnostic tests
- Suitable treatment or remedies
- Whether immediate medical attention is needed
Please avoid disclaimers.
"""

    encoded_image = encode_image(image_filepath) if image_filepath else None
    response = analyze_image_with_query(
        query=full_prompt,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        encoded_image=encoded_image,
        chat_history=chat_history
    )

    chat_history.append((user_query, response))
    conversation_log.append(f"User: {user_query}\nDoctor: {response}\n")

    if check_emergency(response):
        response += "\n\n⚠️ Please seek immediate medical attention!"

    clean_response = re.sub(r'\*+', '', response)  # 🔧 Remove asterisks
    audio_out = "consultation.mp3"
    text_to_speech_with_gtts(clean_response, audio_out)

    return user_query, clean_response, audio_out  # ⬅️ Send clean version to UI

def followup_question(prev_response, new_query):
    if not new_query:
        return "Please enter a follow-up question.", "", None

    retrieved_context = retrieve_context(new_query)

    full_query = f"""
{system_prompt}

Previous Response:
{prev_response}

User Follow-up:
{new_query}

Additional Relevant Medical Knowledge:
{retrieved_context}

Update your medical advice accordingly.
"""

    response = analyze_image_with_query(
        query=full_query,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        encoded_image=None,
        chat_history=chat_history
    )

    chat_history.append((new_query, response))
    conversation_log.append(f"User(Follow-up): {new_query}\nDoctor: {response}\n")

    if check_emergency(response):
        response += "\n\n⚠️ Follow-up indicates possible emergency. Seek medical help!"

    clean_response = re.sub(r'\*+', '', response)  # 🔧 Clean again
    audio_out = "followup.mp3"
    text_to_speech_with_gtts(clean_response, audio_out)

    return new_query, clean_response, audio_out  # ⬅️ Send clean version

def clear_all():
    global chat_history, conversation_log
    chat_history = []
    conversation_log = []
    return None, "", None, "", "", "", None

def download_history():
    path = "consultation_history.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_log))
    return path

def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🩺🤖 MediBot - AI Doctor with RAG, Vision, and Voice")
        gr.Markdown("Consult using voice or text. Upload an optional image. Follow-up supported!")

        with gr.Row(equal_height=True):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak Symptoms (Optional)", scale=1)
            manual_text_input = gr.Textbox(lines=5, label="🖋️ Or Type Symptoms (Optional)", scale=1)
            image_input = gr.Image(type="filepath", label="🖼️ Upload Medical Image (Optional)", scale=1)

        consult_btn = gr.Button("Consult Doctor")

        # 🔔 Blinking CSS + JS
        with gr.Row():
            gr.HTML("""
                <style>
                    .blinking-btn {
                        animation: blink 1s infinite;
                        font-weight: bold;
                        background-color: #ffcc00 !important;
                    }
                    @keyframes blink {
                        0% { opacity: 1; }
                        50% { opacity: 0.4; }
                        100% { opacity: 1; }
                    }
                </style>
                <script>
                    document.addEventListener("DOMContentLoaded", function() {
                        const buttons = document.querySelectorAll('button');
                        buttons.forEach(btn => {
                            if (btn.innerText.includes("Consult Doctor")) {
                                btn.classList.add("blinking-btn");
                            }
                        });
                    });
                </script>
            """)

        speech_out = gr.Textbox(label="📝 Your Query")
        doctor_resp = gr.Textbox(label="🧑‍⚕️ Doctor's Response")
        audio_out = gr.Audio(type="filepath", label="🔊 Doctor Speaking")

        gr.Markdown("---")
        gr.Markdown("## Follow-up Question")
        followup_text = gr.Textbox(label="💬 Type Follow-up Query")
        followup_btn = gr.Button("Submit Follow-up")

        clear_btn = gr.Button("Clear All")
        download_btn = gr.Button("Download Consultation History")
        history_file = gr.File()

        consult_btn.click(
            consult_doctor,
            inputs=[audio_input, manual_text_input, image_input],
            outputs=[speech_out, doctor_resp, audio_out]
        )

        followup_btn.click(
            followup_question,
            inputs=[doctor_resp, followup_text],
            outputs=[followup_text, doctor_resp, audio_out]
        )

        clear_btn.click(
            clear_all,
            inputs=[],
            outputs=[audio_input, manual_text_input, image_input, followup_text, speech_out, doctor_resp, audio_out]
        )

        download_btn.click(
            download_history,
            inputs=[],
            outputs=[history_file]
        )

    demo.launch(debug=True)

if __name__ == "__main__":
    launch_interface()
