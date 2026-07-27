import os
from dotenv import load_dotenv
load_dotenv()

from pyngrok import ngrok

# Set your authtoken 
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Open a tunnel to the Streamlit app running on port 8501
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")

input("Press Enter to stop the tunnel...\n")
ngrok.disconnect(public_url)