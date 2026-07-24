import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MAX_MESSAGES = 8

chat_history = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers short."}
]

def estimate_tokens(messages):
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)

def trim_history():
    """Sirf system prompt + aakhri MAX_MESSAGES rakho, purane hata do"""
    global chat_history
    system_msg = chat_history[0]
    recent_msgs = chat_history[1:]
    
    if len(recent_msgs) > MAX_MESSAGES:
        recent_msgs = recent_msgs[-MAX_MESSAGES:]
    
    chat_history = [system_msg] + recent_msgs

def send_message(user_text):
    chat_history.append({"role": "user", "content": user_text})

    trim_history()
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=chat_history
    )
    
    bot_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": bot_reply})
    trim_history()
    
    tokens = estimate_tokens(chat_history)
    print(f"[Messages in history: {len(chat_history)} | Estimated tokens: {tokens}]")
    
    return bot_reply

print("Buffer memory chat shuru! 'quit' likh ke exit karo.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    reply = send_message(user_input)
    print(f"Bot: {reply}\n")