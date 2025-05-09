from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import os

client = OpenAI()

SYSTEM_PROMPT = (
    "Eres un asistente amable y conciso. Responde en un máximo de 3-4 líneas "
    "y en español neutro. Si no sabes la respuesta, reconócelo brevemente."
)

def generate_reply(user_msg: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp(req: Request):
    form = await req.form()
    incoming = form.get("Body", "")
    answer = generate_reply(incoming)
    twi_resp = MessagingResponse()
    twi_resp.message(answer)
    return Response(str(twi_resp), media_type="application/xml")

@app.get("/")                # health-check opcional
def root():
    return {"status": "ok"}
