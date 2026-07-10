import os
import cv2
import json
import time

from utils.image_processing import load_image, resize_image
from utils.helper import read_prompt

from utils.gemini_api import (
    configure_gemini,
    extract_prescription_json
)


def process_single_image(image_path, client, prompt, output_dir="output"):

    filename = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, filename + ".json")

    if os.path.exists(json_path):
        print(f"Skipping {os.path.basename(image_path)} (already processed)")
        return "skipped"

    print("\n" + "=" * 60)
    print(f"Processing : {os.path.basename(image_path)}")
    print("=" * 60)

    image = load_image(image_path)

    if image is None:
        print("Image not found.")
        return "failed"

    resized = resize_image(image)

    prescription = extract_prescription_json(client, resized, prompt)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(prescription, file, indent=4, ensure_ascii=False)

    print(f"Saved : {json_path}")

    return "success"


def main():

    start = time.time()

    print("Connecting to Gemini...")
    client = configure_gemini()
    print("Connected Successfully!")

    prompt = read_prompt("prompts/extraction_prompt.txt")

    images_folder = "images"
    os.makedirs("output", exist_ok=True)

    image_files = os.listdir(images_folder)

    total = 0
    success = 0
    skipped = 0

    for file in image_files:

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total += 1
        image_path = os.path.join(images_folder, file)
        image_start = time.time()

        try:
            result = process_single_image(image_path, client, prompt)

            if result == "success":
                success += 1
            elif result == "skipped":
                skipped += 1

        except Exception as e:
            print(f"\nError processing {file}")
            print(e)

        image_end = time.time()
        print(f"Time : {image_end-image_start:.2f} sec")

    end = time.time()

    print("\n" + "=" * 60)
    print("Batch Processing Completed")
    print("=" * 60)
    print(f"Images Processed : {total}")
    print(f"Successful (new) : {success}")
    print(f"Skipped (cached) : {skipped}")
    print(f"Failed           : {total - success - skipped}")
    print(f"Total Time       : {end-start:.2f} sec")


if __name__ == "__main__":
    main()