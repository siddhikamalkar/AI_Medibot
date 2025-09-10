# import os
# import base64
# from groq import Groq
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# # Step 2: Convert image to required format
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# # Step 3: Setup Multimodal LLM (with history)
# def analyze_image_with_query(query, model, encoded_image=None, chat_history=[]):
#     client = Groq(api_key=GROQ_API_KEY)

#     # Prepare messages
#     messages = []

#     # Add past history if available
#     for user_query, bot_reply in chat_history[-2:]:  # Only last 2 turns
#         messages.append({"role": "user", "content": user_query})
#         messages.append({"role": "assistant", "content": bot_reply})

#     # Add current user message
#     user_message = {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": query}
#         ]
#     }

#     if encoded_image:
#         user_message["content"].append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{encoded_image}",
#             },
#         })

#     messages.append(user_message)

#     # Call Groq API
#     chat_completion = client.chat.completions.create(
#         messages=messages,
#         model=model
#     )

#     return chat_completion.choices[0].message.content

import os
import base64
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Encode image into base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Analyze image + query with LLM
def analyze_image_with_query(query, model, encoded_image=None, chat_history=[]):
    client = Groq(api_key=GROQ_API_KEY)

    messages = []

    # Add past history if available
    for user_query, bot_reply in chat_history[-2:]:  # Only last 2 turns
        messages.append({"role": "user", "content": user_query})
        messages.append({"role": "assistant", "content": bot_reply})

    # Prepare current user message
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": query}
        ]
    }

    if encoded_image:
        user_message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
            },
        })

    # 🔵 Print full prompt being sent
    print("\n📜 FINAL QUERY SENT TO MODEL:\n")
    print(query)
    print("\n" + "-"*80 + "\n")

    messages.append(user_message)

    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

