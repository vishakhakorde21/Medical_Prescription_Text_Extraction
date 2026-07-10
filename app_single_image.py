import cv2
import json

from utils.image_processing import (
    load_image,
    resize_image,
    convert_to_grayscale,
    blur_image,
    adaptive_threshold,
    remove_noise,
    morphological_processing
)

from utils.helper import read_prompt

from utils.gemini_api import (
    configure_gemini,
    extract_prescription_json
)


def main():

    # -----------------------------
    # Step 1: Load Image
    # -----------------------------
    image_path = "images/prescription1.jpg"

    image = load_image(image_path)

    if image is None:
        print("Image not found!")
        return

    # -----------------------------
    # Step 2: Resize Image
    # -----------------------------
    resized = resize_image(image)

    # -----------------------------
    # Step 3: Convert to Grayscale
    # -----------------------------
    gray = convert_to_grayscale(resized)

    # -----------------------------
    # Step 4: Blur Image
    # -----------------------------
    blurred = blur_image(gray)

    # -----------------------------
    # Step 5: Adaptive Threshold
    # -----------------------------
    threshold = adaptive_threshold(blurred)

    # -----------------------------
    # Step 6: Remove Noise
    # -----------------------------
    denoised = remove_noise(threshold)

    # -----------------------------
    # Step 7: Morphological Processing
    # -----------------------------
    morph = morphological_processing(denoised)

    print("=" * 40)
    print("Image Processing Completed")
    print("=" * 40)

    print("Original Shape :", image.shape)
    print("Resized Shape  :", resized.shape)
    print("Gray Shape     :", gray.shape)
    print("Blurred Shape  :", blurred.shape)
    print("Threshold Shape:", threshold.shape)
    print("Denoised Shape :", denoised.shape)
    print("Morph Shape    :", morph.shape)

    # -----------------------------
    # Step 8: Save Processed Images
    # -----------------------------
    cv2.imwrite("output/resized.jpg", resized)
    cv2.imwrite("output/grayscale.jpg", gray)
    cv2.imwrite("output/blurred.jpg", blurred)
    cv2.imwrite("output/threshold.jpg", threshold)
    cv2.imwrite("output/denoised.jpg", denoised)
    cv2.imwrite("output/final_processed.jpg", morph)

    print("\nAll processed images saved inside the output folder.")

    # -----------------------------
    # Step 9: Connect to Gemini
    # -----------------------------
    print("\nConnecting to Gemini...")

    client = configure_gemini()

    print("Successfully connected to Gemini!")

    # -----------------------------
    # Step 10: Read Prompt
    # -----------------------------
    prompt = read_prompt("prompts/extraction_prompt.txt")

    # -----------------------------
    # Step 11: Extract Structured JSON
    # -----------------------------
    print("\nExtracting structured prescription data...")

    prescription_data = extract_prescription_json(
        client,
        "output/final_processed.jpg",
        prompt
    )

    # -----------------------------
    # Step 12: Display JSON
    # -----------------------------
    print("\n========== Extracted Prescription JSON ==========\n")

    print(
        json.dumps(
            prescription_data,
            indent=4,
            ensure_ascii=False
        )
    )

    # -----------------------------
    # Step 13: Save JSON
    # -----------------------------
    with open(
        "output/prescription.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            prescription_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nPrescription JSON saved successfully!")
    print("Program finished successfully.")


if __name__ == "__main__":
    main()