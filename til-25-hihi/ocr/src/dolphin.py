""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

import cv2
from omegaconf import OmegaConf
from PIL import Image

from chat import DOLPHIN
from utils.utils import *


def process_page(image_path, model, save_dir, max_batch_size):
    """Parse document images with two stages"""
    # Stage 1: Page-level layout and reading order parsing
    pil_image = Image.open(image_path).convert("RGB")
    layout_output = model.chat("Parse the reading order of this document.", pil_image)

    # Stage 2: Element-level content parsing
    padded_image, dims = prepare_image(pil_image)
    recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size)

    # Save outputs
    json_path = save_outputs(recognition_results, image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    text_table_elements = []  # Elements that need processing
    figure_results = []  # Figure elements (no processing needed)
    previous_box = None
    reading_order = 0

    # Collect elements for processing
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    # For figure regions, add empty text result immediately
                    figure_results.append(
                        {
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": "",
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # For text or table regions, prepare for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    prompt = "Parse the table in the image." if label == "tab" else "Read text in the image."
                    text_table_elements.append(
                        {
                            "crop": pil_crop,
                            "prompt": prompt,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Parse text/table elements in parallel
    recognition_results = figure_results
    if text_table_elements:
        crops_list = [elem["crop"] for elem in text_table_elements]
        prompts_list = [elem["prompt"] for elem in text_table_elements]

        # Inference in batch
        batch_results = model.chat(prompts_list, crops_list, max_batch_size=max_batch_size)

        # Add batch results to recognition_results
        for i, result in enumerate(batch_results):
            elem = text_table_elements[i]
            recognition_results.append(
                {
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                }
            )

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def main():
    parser = argparse.ArgumentParser(description="Document processing tool using DOLPHIN model")
    parser.add_argument("--config", default="./config/Dolphin.yaml", help="Path to configuration file")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image or directory of images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=4,
        help="Maximum number of document elements to parse in a single batch (default: 4)",
    )
    args = parser.parse_args()

    # Load Model
    config = OmegaConf.load(args.config)
    model = DOLPHIN(config)

    # Collect Document Images
    if os.path.isdir(args.input_path):
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            image_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        image_files = sorted(image_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        image_files = [args.input_path]

    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    setup_output_dirs(save_dir)

    total_samples = len(image_files)
    print(f"\nTotal samples to process: {total_samples}")

    # Process All Document Images
    for image_path in image_files:
        print(f"\nProcessing {image_path}")
        try:
            json_path, recognition_results = process_page(
                image_path=image_path,
                model=model,
                save_dir=save_dir,
                max_batch_size=args.max_batch_size,
            )

            print(f"Processing completed. Results saved to {save_dir}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()