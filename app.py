"""
app.py — WhatsApp bot con:
• Router (ventas / soporte / general)
• Memoria por usuario (Redis o RAM)
• RAG sobre docs propios (FAQ + catálogo)
• Logs DEBUG/INFO/ERROR

Endpoints
    POST /whatsapp   ← Twilio envía cada mensaje
    GET  /           ← health-check
"""

# ───────────────────────── Imports ─────────────────────────
import os, sys, logging, pathlib
from dotenv import load_dotenv; load_dotenv()

# FastAPI / Twilio
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

# LangChain – modelos y utilidades
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory, ChatMessageHistory
)

# LangChain – RAG
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter

# ───────────────────────── Logger ─────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("whatsapp-bot")

# ───────────────────────── Modelos OpenAI ─────────────────────────
ROUTER_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)   # clasificador
WORKER_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)   # agentes/RAG

# ───────────────────────── Vector-store (RAG) ─────────────────────────
def load_vector_db() -> Chroma:
    """Crea o abre un Chroma persistido en ./rag_chroma."""
    if pathlib.Path("rag_chroma").exists():
        return Chroma(
            persist_directory="rag_chroma",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )

    docs_path = pathlib.Path("docs")
    raw_docs = TextLoader(docs_path / "faq.txt").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    vectordb = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="rag_chroma",
    )
    vectordb.persist()
    logger.info("Vector-store creado con %d chunks", len(chunks))
    return vectordb

VECTOR_DB = load_vector_db()
RETRIEVER = VECTOR_DB.as_retriever(search_kwargs={"k": 3})

def docs2str(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt_rag = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Responde solo con la información del contexto. "
         "Si no la encuentras, responde: No lo sé."),
        ("system", "Contexto:\n{context}\n----------------"),
        ("human", "{question}")
    ]
)

RAG_CHAIN = (
    {
        "context": itemgetter("question") | RETRIEVER | RunnableLambda(docs2str),
        "question": itemgetter("question"),
    }
    | prompt_rag
    | WORKER_LLM
    | StrOutputParser()
)

# ───────────────────────── Agentes especializados ─────────────────────────
def make_agent_prompt(sys_msg: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", sys_msg),
         MessagesPlaceholder("history"),
         ("human", "{input}")]
    )

def make_chain(prompt) -> RunnableLambda:
    return prompt | WORKER_LLM | StrOutputParser()

SALES_CHAIN = make_chain(make_agent_prompt(
    "Eres un representante de ventas entusiasta. "
    "Responde sobre precios, promociones o stock.")
)
SUPPORT_CHAIN = make_chain(make_agent_prompt(
    "Eres agente de soporte técnico. Sé empático y claro.")
)
GENERAL_FALLBACK = make_chain(make_agent_prompt(
    "Eres un asistente general, amable y conciso.")
)

AGENTS = {
    "ventas": SALES_CHAIN,
    "soporte": SUPPORT_CHAIN,
    "general_fallback": GENERAL_FALLBACK,  # se usa si RAG falla
}

def answer_general(data: dict) -> str:
    """RAG + fallback al asistente general."""
    rag_resp = RAG_CHAIN.invoke({"question": data["input"]})
    if "no lo sé" in rag_resp.lower():
        return AGENTS["general_fallback"].invoke(data)
    return rag_resp

AGENTS["general"] = answer_general  # reemplaza el antiguo general

# ───────────────────────── Router con contexto ─────────────────────────
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un clasificador que responde SOLO con ventas, soporte o general."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

def classify(data: dict) -> str:
    resp = ROUTER_LLM.invoke(router_prompt.format_messages(
        input=data["input"], history=data["history"]))
    label = resp.content.strip().lower()
    return label if label in {"ventas", "soporte"} else "general"

# ───────────────────────── Memoria por usuario ─────────────────────────
REDIS_URL = os.getenv("REDIS_URL")
CACHE = {}
def history_for(session_id: str):
    if session_id in CACHE:
        return CACHE[session_id]
    hist = (RedisChatMessageHistory(url=REDIS_URL,
            key=f"whatsapp:chat:{session_id}", ttl=60*60*24*7)
            if REDIS_URL else ChatMessageHistory())
    CACHE[session_id] = hist
    return hist

# ───────────────────────── Pipeline principal ─────────────────────────
def generate_reply(phone: str, text: str) -> str:
    hist = history_for(phone)
    hist.add_user_message(text)

    payload = {"input": text, "history": hist.messages}
    agent   = classify(payload)
    logger.info("→ %s | agente=%s", phone, agent)

    handler = AGENTS.get(agent, GENERAL_FALLBACK)

    try:
        # si es función llámala, si es Runnable usa .invoke
        answer = handler(payload) if callable(handler) else handler.invoke(payload)
    except Exception as e:
        logger.error("LLM error: %s", e, exc_info=True)
        answer = "Lo siento, ocurrió un problema interno."

    hist.add_ai_message(answer)
    logger.debug("← %s | resp=%s", phone, answer[:120])
    return answer

# ───────────────────────── FastAPI / Twilio ─────────────────────────
app = FastAPI()

@app.post("/whatsapp")
async def whatsapp(req: Request):
    form  = await req.form()
    body  = form.get("Body", "")
    phone = form.get("From", "unknown")
    logger.info("MSG %s: %s", phone, body[:60])

    reply = generate_reply(phone, body)

    tw = MessagingResponse(); tw.message(reply)
    return Response(str(tw), media_type="application/xml")

@app.get("/")
def root():
    return {"status": "ok"}

# ───────────────────────── Dev server ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.setLevel(logging.DEBUG)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
