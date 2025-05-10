"""
app.py  –  WhatsApp LLM bot con memoria

• POST /whatsapp  ← Twilio llama aquí por cada mensaje
• GET  /          ← health-check
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────
# 1.  MODELOS OPENAI
# ──────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser

LLM_ROUTER = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
LLM_WORKER = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

# ──────────────────────────────────────
# 2.  PROMPTS DE LOS AGENTES
# ──────────────────────────────────────
def make_agent_prompt(role_system: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", role_system),
            MessagesPlaceholder(variable_name="history"),   # historial
            ("human", "{input}"),
        ]
    )

prompt_sales   = make_agent_prompt(
    "Eres un representante de ventas entusiasta. "
    "Responde de forma breve sobre precios, promociones o disponibilidad."
)
prompt_support = make_agent_prompt(
    "Eres agente de soporte técnico. Sé empático y claro."
)
prompt_general = make_agent_prompt(
    "Eres un asistente general, amable y conciso."
)

def make_chain(prompt):
    return prompt | LLM_WORKER | StrOutputParser()

SALES_CHAIN   = make_chain(prompt_sales)
SUPPORT_CHAIN = make_chain(prompt_support)
GENERAL_CHAIN = make_chain(prompt_general)

# ──────────────────────────────────────
# 3.  ROUTER (clasificador)
# ──────────────────────────────────────
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un clasificador que responde SOLO con la palabra "
         "`ventas`, `soporte` o `general`.\n\n"
         "Reglas:\n"
         "- Ventas: precios, promociones, disponibilidad.\n"
         "- Soporte: quejas, problemas técnicos, devoluciones.\n"
         "- General: todo lo demás."),
        ("human", "{input}"),
    ]
)

def route(message: str) -> str:
    resp = LLM_ROUTER.invoke(router_prompt.format_messages(input=message))
    return resp.content.strip().lower()

def router_fn(data: dict) -> str:
    """
    data = { "input": <texto>, "history": [...] }
    """
    choice = route(data["input"])
    if choice == "ventas":
        return SALES_CHAIN.invoke(data)
    elif choice == "soporte":
        return SUPPORT_CHAIN.invoke(data)
    return GENERAL_CHAIN.invoke(data)

# ──────────────────────────────────────
# 4.  MEMORIA POR USUARIO
# ──────────────────────────────────────
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory,
    ChatMessageHistory,
)

REDIS_URL   = os.getenv("REDIS_URL")         # Railway plugin o None
KEY_PREFIX  = "whatsapp:chat:"               # clave base en Redis
_history_cache: dict[str, ChatMessageHistory] = {}   # cache local

def get_history(session_id: str):
    """
    Devuelve SIEMPRE la misma instancia por sesión.
    Usa Redis si hay REDIS_URL, si no ChatMessageHistory en RAM.
    """
    if session_id in _history_cache:
        return _history_cache[session_id]

    if REDIS_URL:
        hist = RedisChatMessageHistory(
            url=REDIS_URL,
            key=f"{KEY_PREFIX}{session_id}",
            ttl=60 * 60 * 24 * 7,     # 7 días
        )
    else:
        hist = ChatMessageHistory()

    _history_cache[session_id] = hist
    return hist

def generate_reply(phone: str, user_msg: str) -> str:
    history = get_history(phone)                 # instancia única

    history.add_user_message(user_msg)           # guarda turno usuario

    router_runnable = RunnableLambda(router_fn)

    chain_with_history = RunnableWithMessageHistory(
        router_runnable,
        lambda _: history,                       # MISMA instancia
        input_messages_key="input",
        history_messages_key="history",
    )

    answer = chain_with_history.invoke(
        {"input": user_msg},
        {"configurable": {"session_id": phone}},
    )

    history.add_ai_message(answer)               # guarda respuesta
    return answer

# ──────────────────────────────────────
# 5.  API FastAPI + Twilio
# ──────────────────────────────────────
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp_webhook(req: Request):
    form  = await req.form()
    body  = form.get("Body", "")
    phone = form.get("From", "unknown")

    try:
        reply = generate_reply(phone, body)
    except Exception as e:
        print("🛑 INTERNAL ERROR:", e)
        reply = "Lo siento, ocurrió un problema interno."

    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(str(twiml), media_type="application/xml")

@app.get("/")       # health-check
def root():
    return {"status": "ok"}

# ──────────────────────────────────────
# 6.  EJECUCIÓN LOCAL
# ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
