# # import os
# # import json
# # import time

# # from dotenv import load_dotenv
# # from google import genai
# # from google.genai.errors import ServerError
# # from PIL import Image


# # def configure_gemini():
# #     """
# #     Loads the Gemini API key from the .env file
# #     and creates a Gemini client.

# #     Returns:
# #         Gemini Client object.
# #     """

# #     # Load environment variables
# #     load_dotenv()

# #     # Read API key
# #     api_key = os.getenv("GOOGLE_API_KEY")

# #     # Validate API key
# #     if not api_key:
# #         raise ValueError("GOOGLE_API_KEY not found in .env file.")

# #     # Create Gemini client
# #     client = genai.Client(api_key=api_key)

# #     return client


# # def extract_prescription_json(client, image_path, prompt):
# #     """
# #     Sends the processed prescription image to Gemini
# #     and returns structured JSON.

# #     Parameters:
# #         client : Gemini Client
# #         image_path (str)
# #         prompt (str)

# #     Returns:
# #         dict
# #     """

# #     # Open image
# #     image = Image.open(image_path)

# #     max_retries = 3

# #     for attempt in range(max_retries):

# #         try:

# #             response = client.models.generate_content(
# #                 model="gemini-2.5-flash",
# #                 contents=[
# #                     prompt,
# #                     image
# #                 ]
# #             )

# #             # -----------------------------
# #             # Debug: Print Gemini Response
# #             # -----------------------------
# #             print("\n" + "=" * 60)
# #             print("RAW GEMINI RESPONSE")
# #             print("=" * 60)
# #             print(response.text)
# #             print("=" * 60 + "\n")

# #             # -----------------------------
# #             # Clean Markdown if Gemini
# #             # returns ```json ... ```
# #             # -----------------------------
# #             response_text = response.text.strip()

# #             if response_text.startswith("```json"):
# #                 response_text = response_text.replace(
# #                     "```json",
# #                     "",
# #                     1
# #                 )

# #             if response_text.startswith("```"):
# #                 response_text = response_text.replace(
# #                     "```",
# #                     "",
# #                     1
# #                 )

# #             if response_text.endswith("```"):
# #                 response_text = response_text[:-3]

# #             response_text = response_text.strip()

# #             # -----------------------------
# #             # Convert JSON string
# #             # into Python dictionary
# #             # -----------------------------
# #             json_data = json.loads(response_text)

# #             return json_data

# #         except json.JSONDecodeError as e:

# #             print("\nInvalid JSON received from Gemini.")
# #             print(e)

# #             raise Exception(
# #                 "Gemini did not return valid JSON."
# #             )

# #         except ServerError:

# #             print("\nGemini server is busy.")
# #             print(f"Retrying... ({attempt + 1}/{max_retries})")

# #             time.sleep(5)

# #     raise Exception(
# #         "Gemini server is unavailable. Please try again later."
# #     )
##**************************************************
# import os
# import json
# import time
# import cv2

# from dotenv import load_dotenv
# from google import genai
# from google.genai.errors import ServerError
# from PIL import Image


# def configure_gemini():
#     """
#     Loads the Gemini API key from the .env file
#     and creates a Gemini client.
#     """

#     load_dotenv()

#     api_key = os.getenv("GOOGLE_API_KEY")

#     if not api_key:
#         raise ValueError("GOOGLE_API_KEY not found.")

#     client = genai.Client(api_key=api_key)

#     return client


# def extract_prescription_json(client, processed_image, prompt):
#     """
#     Sends the processed OpenCV image directly to Gemini.

#     Parameters:
#         client
#         processed_image (numpy.ndarray)
#         prompt (str)

#     Returns:
#         dict
#     """

#     # Convert OpenCV (BGR) → RGB
#     rgb = cv2.cvtColor(
#         processed_image,
#         cv2.COLOR_BGR2RGB
#     )

#     # Convert NumPy → PIL
#     image = Image.fromarray(rgb)

#     max_retries = 3

#     for attempt in range(max_retries):

#         try:

#             response = client.models.generate_content(
#                 model="gemini-2.5-flash",
#                 contents=[
#                     prompt,
#                     image
#                 ]
#             )

#             print("\n" + "=" * 60)
#             print("RAW GEMINI RESPONSE")
#             print("=" * 60)
#             print(response.text)
#             print("=" * 60)

#             text = response.text.strip()

#             # Remove Markdown if Gemini returns it
#             if text.startswith("```json"):
#                 text = text.replace("```json", "", 1)

#             if text.endswith("```"):
#                 text = text[:-3]

#             text = text.strip()

#             json_data = json.loads(text)

#             return json_data

#         except json.JSONDecodeError as e:

#             print("\nInvalid JSON returned.")
#             print(e)

#             raise Exception(
#                 "Gemini returned invalid JSON."
#             )

#         except ServerError:

#             print(
#                 f"\nGemini busy...Retry {attempt+1}/{max_retries}"
#             )

#             time.sleep(5)

#     raise Exception(
#         "Gemini server unavailable."
#     )
#*************************************************

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