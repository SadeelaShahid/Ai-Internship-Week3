import os
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Model load ho raha hai...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

with open("day15/sample_document.txt", "r", encoding="utf-8") as f:
    document_text = f.read()

chunks = chunk_text(document_text, chunk_size=100, overlap=20)
print(f"{len(chunks)} chunks banay gaye.\n")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_v2_docs")

embeddings = embed_model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, embeddings=embeddings, ids=ids)

def retrieve_top_k(query, k=3):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results['documents'][0]

MAX_MESSAGES = 8
conversation_history = []

def trim_history():
    global conversation_history
    if len(conversation_history) > MAX_MESSAGES:
        conversation_history = conversation_history[-MAX_MESSAGES:]

def rewrite_query(history, new_question):
    if not history:
        return new_question
    
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = f"""Given this conversation history:
{history_text}

And this new follow-up question: "{new_question}"

Rewrite the follow-up question to be a standalone question. If it's already standalone, return it as-is. Only output the rewritten question, nothing else.

Rewritten question:"""

    response = client_llm.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def ask_rag(question):

    standalone_question = rewrite_query(conversation_history, question)

    top_chunks = retrieve_top_k(standalone_question, k=3)
    context = "\n\n".join(top_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""
    
    response = client_llm.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    trim_history()
    
    return answer, standalone_question

print("Conversational RAG chatbot shuru! 'quit' likh ke exit karo.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    answer, rewritten = ask_rag(user_input)
    if rewritten != user_input:
        print(f"[Rewritten as: {rewritten}]")
    print(f"Bot: {answer}\n")