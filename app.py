# # import os
# # import cv2
# # import json
# # import time

# # from utils.image_processing import (
# #     load_image,
# #     resize_image,
# #     convert_to_grayscale,
# #     blur_image,
# #     adaptive_threshold,
# #     remove_noise,
# #     morphological_processing
# # )

# # from utils.helper import read_prompt

# # from utils.gemini_api import (
# #     configure_gemini,
# #     extract_prescription_json
# # )


# # def process_single_image(image_path, client, prompt):
# #     """
# #     Processes one prescription image and
# #     saves the processed image and JSON.
# #     """

# #     print("\n" + "=" * 60)
# #     print(f"Processing : {os.path.basename(image_path)}")
# #     print("=" * 60)

# #     image = load_image(image_path)

# #     if image is None:
# #         print("Could not load image.")
# #         return False

# #     # -----------------------------
# #     # Image Processing
# #     # -----------------------------
# #     resized = resize_image(image)

# #     gray = convert_to_grayscale(resized)

# #     blurred = blur_image(gray)

# #     threshold = adaptive_threshold(blurred)

# #     denoised = remove_noise(threshold)

# #     morph = morphological_processing(denoised)

# #     # -----------------------------
# #     # Save Processed Image
# #     # -----------------------------
# #     filename = os.path.splitext(
# #         os.path.basename(image_path)
# #     )[0]

# #     processed_image_path = os.path.join(
# #         "output",
# #         filename + "_processed.jpg"
# #     )

# #     cv2.imwrite(
# #         processed_image_path,
# #         morph
# #     )

# #     print("Processed image saved.")

# #     # -----------------------------
# #     # Gemini Extraction
# #     # -----------------------------
# #     prescription_data = extract_prescription_json(
# #         client,
# #         processed_image_path,
# #         prompt
# #     )

# #     # -----------------------------
# #     # Save JSON
# #     # -----------------------------
# #     json_path = os.path.join(
# #         "output",
# #         filename + ".json"
# #     )

# #     with open(
# #         json_path,
# #         "w",
# #         encoding="utf-8"
# #     ) as file:

# #         json.dump(
# #             prescription_data,
# #             file,
# #             indent=4,
# #             ensure_ascii=False
# #         )

# #     print(f"JSON saved : {json_path}")

# #     return True


# # def main():

# #     start_time = time.time()

# #     # -----------------------------
# #     # Configure Gemini
# #     # -----------------------------
# #     print("Connecting to Gemini...")

# #     client = configure_gemini()

# #     print("Connected successfully!")

# #     # -----------------------------
# #     # Read Prompt
# #     # -----------------------------
# #     prompt = read_prompt(
# #         "prompts/extraction_prompt.txt"
# #     )

# #     # -----------------------------
# #     # Images Folder
# #     # -----------------------------
# #     images_folder = "images"

# #     image_files = os.listdir(images_folder)

# #     total_images = 0

# #     successful = 0

# #     # -----------------------------
# #     # Process Every Image
# #     # -----------------------------
# #     for file in image_files:

# #         if not file.lower().endswith(
# #             (".jpg", ".jpeg", ".png")
# #         ):
# #             continue

# #         total_images += 1

# #         image_path = os.path.join(
# #             images_folder,
# #             file
# #         )

# #         image_start = time.time()

# #         success = process_single_image(
# #             image_path,
# #             client,
# #             prompt
# #         )

# #         image_end = time.time()

# #         print(
# #             f"Time Taken : {image_end-image_start:.2f} seconds"
# #         )

# #         if success:
# #             successful += 1

# #     end_time = time.time()

# #     print("\n" + "=" * 60)
# #     print("Batch Processing Completed")
# #     print("=" * 60)

# #     print(f"Total Images     : {total_images}")
# #     print(f"Successful       : {successful}")
# #     print(f"Failed           : {total_images-successful}")
# #     print(f"Total Time       : {end_time-start_time:.2f} sec")

# #     if successful > 0:
# #         print(
# #             f"Average Time/Image : {(end_time-start_time)/successful:.2f} sec"
# #         )

#***************************************************************
# # if __name__ == "__main__":
# #     main()

# import os
# import cv2
# import json
# import time

# from utils.image_processing import (
#     load_image,
#     resize_image,
#     convert_to_grayscale,
#     blur_image,
#     adaptive_threshold,
#     remove_noise,
#     morphological_processing
# )

# from utils.helper import read_prompt

# from utils.gemini_api import (
#     configure_gemini,
#     extract_prescription_json
# )


# def process_single_image(image_path, client, prompt):

#     print("\n" + "=" * 60)
#     print(f"Processing : {os.path.basename(image_path)}")
#     print("=" * 60)

#     image = load_image(image_path)

#     if image is None:
#         print("Image not found.")
#         return False

#     # Image preprocessing
#     resized = resize_image(image)

#     gray = convert_to_grayscale(resized)

#     blurred = blur_image(gray)

#     threshold = adaptive_threshold(blurred)

#     denoised = remove_noise(threshold)

#     morph = morphological_processing(denoised)

#     # Gemini Extraction
#     prescription = extract_prescription_json(
#         client,
#         morph,
#         prompt
#     )

#     filename = os.path.splitext(
#         os.path.basename(image_path)
#     )[0]

#     json_path = os.path.join(
#         "output",
#         filename + ".json"
#     )

#     with open(
#         json_path,
#         "w",
#         encoding="utf-8"
#     ) as file:

#         json.dump(
#             prescription,
#             file,
#             indent=4,
#             ensure_ascii=False
#         )

#     print(f"Saved : {json_path}")

#     return True


# def main():

#     start = time.time()

#     print("Connecting to Gemini...")

#     client = configure_gemini()

#     print("Connected Successfully!")

#     prompt = read_prompt(
#         "prompts/extraction_prompt.txt"
#     )

#     images_folder = "images"

#     image_files = os.listdir(images_folder)

#     total = 0
#     success = 0

#     for file in image_files:

#         if not file.lower().endswith(
#             (".jpg", ".jpeg", ".png")
#         ):
#             continue

#         total += 1

#         image_path = os.path.join(
#             images_folder,
#             file
#         )

#         image_start = time.time()

#         try:

#             ok = process_single_image(
#                 image_path,
#                 client,
#                 prompt
#             )

#             if ok:
#                 success += 1

#         except Exception as e:

#             print(f"\nError processing {file}")
#             print(e)

#         image_end = time.time()

#         print(
#             f"Time : {image_end-image_start:.2f} sec"
#         )

#     end = time.time()

#     print("\n" + "=" * 60)
#     print("Batch Processing Completed")
#     print("=" * 60)

#     print(f"Images Processed : {total}")
#     print(f"Successful      : {success}")
#     print(f"Failed          : {total-success}")
#     print(f"Total Time      : {end-start:.2f} sec")

#     if success > 0:

#         print(
#             f"Average/Image   : {(end-start)/success:.2f} sec"
#         )


# if __name__ == "__main__":
#     main()

#***************************************************

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