import os
import json
import time
import cv2

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from PIL import Image


def configure_gemini():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    return genai.Client(api_key=api_key)


def extract_prescription_json(client, processed_image, prompt):
    """
    Sends the image to Gemini with JSON mode enabled
    and thinking disabled, for lower latency and cost.
    """

    rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            print("\n" + "=" * 60)
            print("RAW GEMINI RESPONSE")
            print("=" * 60)
            print(response.text)
            print("=" * 60)

            json_data = json.loads(response.text)

            return json_data

        except json.JSONDecodeError as e:
            print("\nInvalid JSON returned.")
            print(e)
            raise Exception("Gemini returned invalid JSON.")

        except ServerError:
            print(f"\nGemini busy...Retry {attempt+1}/{max_retries}")
            time.sleep(5)

    raise Exception("Gemini server unavailable.")