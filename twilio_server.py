import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

app = Flask(__name__)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route("/twilio/voice", methods=["POST"])
def voice():
    """
    Handle incoming calls from Twilio and start the AI agent conversation.
    """
    response = VoiceResponse()
    # Greet and start the conversation (for now, just a placeholder)
    response.say("Hi, this is Sarah from Melbourne Fitness Studio. How's it going?", voice="Polly.Joanna", language="en-AU")
    # In production, you would stream audio to/from your agent here
    response.pause(length=2)
    response.say("This is a placeholder for the AI agent conversation.", voice="Polly.Joanna", language="en-AU")
    response.hangup()
    return Response(str(response), mimetype="application/xml")

@app.route("/twilio/outbound", methods=["POST"])
def outbound():
    """
    Initiate an outbound call from the Twilio number to a customer.
    Expects JSON: { "to": "+614..." }
    """
    to_number = request.json.get("to")
    if not to_number:
        return {"error": "Missing 'to' number"}, 400
    call = client.calls.create(
        to=to_number,
        from_=TWILIO_PHONE_NUMBER,
        url=request.url_root.rstrip("/") + "/twilio/voice"
    )
    return {"status": "initiated", "sid": call.sid}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 