import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SUMMARIZE_THRESHOLD = 6  # jab itne messages ho jayein, purane summarize kar do
KEEP_RECENT = 2          # summarize karne ke baad, itne recent raw messages rakho

conversation_summary = ""  # shuru mein koi summary nahi

chat_history = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers short."}
]

def estimate_tokens(messages):
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)

def summarize_old_messages(messages_to_summarize):
    """Purane messages ko ek chhoti summary mein compress karta hai (LLM call)"""
    global conversation_summary
    
    text_to_summarize = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages_to_summarize]
    )
    
    prompt = f"""Summarize this conversation in 2-3 short sentences, keeping any important facts 
(like names, preferences, or specific details mentioned):

{text_to_summarize}

Summary:"""
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}]
    )
    
    new_summary = response.choices[0].message.content.strip()
    
    # Agar pehle se summary hai, purani aur nayi ko mila do
    if conversation_summary:
        conversation_summary = conversation_summary + " " + new_summary
    else:
        conversation_summary = new_summary

def trim_and_summarize():
    """Agar history bohot lambi ho jaye, purane messages summarize karke hata do"""
    global chat_history
    
    system_msg = chat_history[0]
    recent_msgs = chat_history[1:]
    
    if len(recent_msgs) > SUMMARIZE_THRESHOLD:
        # Kitne purane messages summarize karne hain
        messages_to_summarize = recent_msgs[:-KEEP_RECENT]
        messages_to_keep = recent_msgs[-KEEP_RECENT:]
        
        summarize_old_messages(messages_to_summarize)
        
        # System prompt mein summary bhi daal do
        updated_system = {
            "role": "system",
            "content": f"You are a helpful assistant. Keep answers short. Context from earlier in the conversation: {conversation_summary}"
        }
        
        chat_history = [updated_system] + messages_to_keep

def send_message(user_text):
    chat_history.append({"role": "user", "content": user_text})
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=chat_history
    )
    
    bot_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": bot_reply})
    
    trim_and_summarize()
    
    tokens = estimate_tokens(chat_history)
    print(f"[Messages in history: {len(chat_history)} | Estimated tokens: {tokens}]")
    if conversation_summary:
        print(f"[Current summary: {conversation_summary}]")
    
    return bot_reply

print("Summarization memory chat shuru! 'quit' likh ke exit karo.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    reply = send_message(user_input)
    print(f"Bot: {reply}\n")