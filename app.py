from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

import os, json


llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)   # decide el destino
llm_worker = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)   # responde


prompt_sales = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un representante de ventas entusiasta. Contesta unicamente con tu función."),
        ("human", "{input}")
    ]
)

prompt_support = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres agente de soporte técnico. Contesta unicamente con tu función."),
        ("human", "{input}")
    ]
)

prompt_general = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente general, amable y conciso. Contesta unicamente con tu función."),
        ("human", "{input}")
    ]
)

def make_chain(prompt):
    return prompt | llm_worker | StrOutputParser()

sales_chain    = make_chain(prompt_sales)
support_chain  = make_chain(prompt_support)
general_chain  = make_chain(prompt_general)

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un clasificador que responde SOLO con la palabra: "
         "`ventas`, `soporte` o `general`.\n\n"
         "Categoriza la consulta del usuario según estas reglas:\n"
         "- Ventas: preguntas sobre precios, promociones, disponibilidades.\n"
         "- Soporte: quejas, problemas técnicos, devoluciones.\n"
         "- General: todo lo demás.\n"),
        ("human", "{input}")
    ]
)

def route(message: str) -> str:
    resp = llm_router.invoke(router_prompt.format_messages(input=message))
    return resp.content.strip().lower()

def router_chain(message: str):
    choice = route(message)
    if choice == "ventas":
        return sales_chain.invoke({"input": message})
    elif choice == "soporte":
        return support_chain.invoke({"input": message})
    else:
        return general_chain.invoke({"input": message})



app = FastAPI()

@app.post("/whatsapp")
async def whatsapp(req: Request):
    form = await req.form()
    incoming = form.get("Body", "")
    try:
        answer = router_chain(incoming)
    except Exception as e:
        print("🛑 Error:", e)
        answer = "Lo siento, ocurrió un problema al procesar tu mensaje."
    twi_resp = MessagingResponse()
    twi_resp.message(answer)
    return Response(str(twi_resp), media_type="application/xml")

@app.get("/")                # health-check opcional
def root():
    return {"status": "ok"}
