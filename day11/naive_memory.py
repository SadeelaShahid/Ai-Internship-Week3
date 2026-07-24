import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

chat_history = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers short."}
]

def estimate_tokens(messages):
    """Rough token estimate: total words * 1.3 (guide ka suggested formula)"""
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)

def send_message(user_text):

    chat_history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=chat_history
    )
    
    bot_reply = response.choices[0].message.content

    chat_history.append({"role": "assistant", "content": bot_reply})

    tokens = estimate_tokens(chat_history)
    print(f"[Total messages in history: {len(chat_history)} | Estimated tokens: {tokens}]")
    
    return bot_reply

print("Chat shuru! 'quit' likh ke exit karo.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    reply = send_message(user_input)
    print(f"Bot: {reply}\n")