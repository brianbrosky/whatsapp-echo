"""
app.py — WhatsApp LLM bot con memoria + logs
"""

# ───────────────── Imports ──────────────────
import os, sys, logging, json
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory, ChatMessageHistory
)

# ───────────── Config. entorno ─────────────
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# ───────────── Logger sencillo ─────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("whatsapp-bot")

# ───────────── Modelos OpenAI ──────────────
ROUTER_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # ← CORREGIDO
WORKER_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

# ───────────── Agentes (prompts + cadenas) ─────────────
def make_agent_prompt(system_msg: str):
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

def make_chain(prompt):
    return prompt | WORKER_LLM | StrOutputParser()

AGENTS = {
    "ventas":  make_chain(make_agent_prompt("Eres agente de ventas.")),
    "soporte": make_chain(make_agent_prompt("Eres agente de soporte.")),
    "general": make_chain(make_agent_prompt("Eres asistente general.")),
}

# ───────────── Router con contexto ─────────────
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un clasificador que responde SOLO con ventas, soporte o general."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

def classify(data: dict) -> str:
    msgs  = router_prompt.format_messages(input=data["input"], history=data["history"])
    resp  = ROUTER_LLM.invoke(msgs)
    label = resp.content.strip().lower()
    return label if label in AGENTS else "general"

# ───────────── Memoria por usuario ─────────────
CACHE = {}
def history_for(session_id: str):
    if session_id in CACHE:
        return CACHE[session_id]
    hist = (RedisChatMessageHistory(
                url=REDIS_URL,
                key=f"whatsapp:chat:{session_id}",
                ttl=60*60*24*7)
            if REDIS_URL else ChatMessageHistory())
    CACHE[session_id] = hist
    return hist

# ───────────── Pipeline principal ─────────────
def generate_reply(phone: str, text: str) -> str:
    hist = history_for(phone)
    hist.add_user_message(text)

    payload = {"input": text, "history": hist.messages}
    agent   = classify(payload)
    logger.info("→ %s | agente=%s", phone, agent)

    try:
        answer = AGENTS[agent].invoke(payload)
    except Exception as e:
        logger.error("LLM error: %s", e, exc_info=True)
        answer = "Lo siento, ocurrió un problema interno."

    #⬇️  NUEVA LÍNEA: muestra la respuesta generada (recorta a 120 chars)
    logger.info("← %s | respuesta=%s", phone, answer[:120])


    hist.add_ai_message(answer)
    return answer

# ───────────── FastAPI / Twilio ─────────────
app = FastAPI()

@app.post("/whatsapp")
async def whatsapp(req: Request):
    form  = await req.form()
    body  = form.get("Body", "")
    phone = form.get("From", "unknown")
    logger.info("MSG from %s: %s", phone, body[:60])

    reply = generate_reply(phone, body)

    tw = MessagingResponse(); tw.message(reply)
    return Response(str(tw), media_type="application/xml")

@app.get("/")
def root():
    return {"status": "ok"}

# ───────────── Ejecución local ─────────────
if __name__ == "__main__":
    import uvicorn
    logger.setLevel(logging.DEBUG)  # más verboso en local
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
