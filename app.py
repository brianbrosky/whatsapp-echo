from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp(req: Request):
    form = await req.form()
    msg  = form.get("Body", "")
    resp = MessagingResponse()
    resp.message(msg)          # eco
    return Response(str(resp), media_type="application/xml")
