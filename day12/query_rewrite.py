import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def rewrite_query(chat_history, new_question):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    
    prompt = f"""Given this conversation history:
{history_text}

And this new follow-up question: "{new_question}"

Rewrite the follow-up question to be a standalone question that makes sense without needing the conversation history. Only output the rewritten question, nothing else.

Rewritten question:"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()


fake_history = [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "You can get a full refund within 30 days of purchase."},
]

follow_up_1 = "What about for digital products?"
rewritten_1 = rewrite_query(fake_history, follow_up_1)
print(f"Original follow-up: {follow_up_1}")
print(f"Rewritten: {rewritten_1}\n")

follow_up_2 = "And the second one?"
rewritten_2 = rewrite_query(fake_history, follow_up_2)
print(f"Original follow-up: {follow_up_2}")
print(f"Rewritten: {rewritten_2}\n")

follow_up_3 = "Why though?"
rewritten_3 = rewrite_query(fake_history, follow_up_3)
print(f"Original follow-up: {follow_up_3}")
print(f"Rewritten: {rewritten_3}\n")