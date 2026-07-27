import os
from dotenv import load_dotenv
load_dotenv()
from pyngrok import ngrok

# Set  authtoken 
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Open a tunnel to the Gradio app running on port 7860
public_url = ngrok.connect(7860)
print(f"Gradio app is live at: {public_url}")

input("Press Enter to stop the tunnel...\n")
ngrok.disconnect(public_url)

