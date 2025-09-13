import cv2
import os
import numpy as np
from bs4 import BeautifulSoup
import re
import random
from PIL import Image # For handling images if cv2 has issues with some formats

def extract_bbox_from_hocr_title(title_str):
    """Extracts x1, y1, x2, y2 from hOCR bbox string."""
    match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title_str)
    if match:
        return [int(c) for c in match.groups()]
    return None

def order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sort points by their y-coordinate
    sorted_y = pts[np.argsort(pts[:, 1]), :]
    
    # Get top two and bottom two points
    top_two = sorted_y[:2, :]
    bottom_two = sorted_y[2:, :]
    
    # Sort top_two by x-coordinate to get top-left and top-right
    top_two = top_two[np.argsort(top_two[:, 0]), :]
    rect[0] = top_two[0] # Top-left
    rect[1] = top_two[1] # Top-right
    
    # Sort bottom_two by x-coordinate (descending for bottom-right, ascending for bottom-left)
    # A simpler way: bottom_two sorted by x will give [potential_bl, potential_br]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0]), :]
    rect[3] = bottom_two[0] # Bottom-left
    rect[2] = bottom_two[1] # Bottom-right
        
    return rect

def get_oriented_line_crop(image, hocr_line_bbox): # hocr_line_bbox is [x1, y1, x2, y2]
    x1_h, y1_h, x2_h, y2_h = hocr_line_bbox

    # Ensure x1 < x2 and y1 < y2
    _x1, _x2 = min(x1_h,x2_h), max(x1_h,x2_h)
    _y1, _y2 = min(y1_h,y2_h), max(y1_h,y2_h)
    x1,y1,x2,y2 = _x1,_y1,_x2,_y2

    if not (x2 > x1 and y2 > y1): # Check for valid box
        print(f"Warning: Invalid hOCR bbox for crop: {hocr_line_bbox}")
        return None
    if image is None:
        print(f"Warning: Input image is None for bbox {hocr_line_bbox}")
        return None
    
    img_h, img_w = image.shape[:2]
    # Clamp coordinates to be within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w -1 , x2)
    y2 = min(img_h -1, y2)

    if not (x2 > x1 and y2 > y1): # Check again after clamping
        print(f"Warning: Invalid hOCR bbox after clamping for crop: orig {hocr_line_bbox}, clamped {[x1,y1,x2,y2]}")
        return None

    rect_pts = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype="float32")

    min_area_rect = cv2.minAreaRect(rect_pts)
    box_cv = cv2.boxPoints(min_area_rect)

    src_pts = order_points(box_cv)

    (tl, tr, br, bl) = src_pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        print(f"Warning: Degenerate minAreaRect (width={max_width}, height={max_height}) for {hocr_line_bbox}. Using simple slice.")
        # Ensure slice indices are valid
        y1_s, y2_s = int(y1), int(y2)
        x1_s, x2_s = int(x1), int(x2)
        if y2_s > y1_s and x2_s > x1_s:
            return image[y1_s:y2_s, x1_s:x2_s]
        else:
            print(f"Error: Simple slice also invalid for {hocr_line_bbox}")
            return None


    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped


# --- Configuration ---
# INPUT_DATA_DIR: Directory containing all .hocr, .jpg, .png, etc. files
INPUT_DATA_DIR = "/home/jupyter/advanced/ocr"

OUTPUT_RECOGNITION_DATA_DIR = "./paddleocr_rec_data_prepared2" # Where cropped images and labels will be saved
TRAIN_CROP_SUBDIR = "train_crops"
VAL_CROP_SUBDIR = "val_crops"

TRAIN_RATIO = 0.8 # 80% for training
# VAL_RATIO will be the remainder (20% in this case if no test set)
# If you want a separate test set, you'll need to adjust VAL_RATIO and add TEST_RATIO logic

# --- Main Script ---
os.makedirs(os.path.join(OUTPUT_RECOGNITION_DATA_DIR, TRAIN_CROP_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_RECOGNITION_DATA_DIR, VAL_CROP_SUBDIR), exist_ok=True)

all_char_set = set()
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

basenames_with_paths = []
print(f"Scanning directory: {INPUT_DATA_DIR}")
for f_name in os.listdir(INPUT_DATA_DIR):
    if f_name.endswith(".hocr"):
        basename = os.path.splitext(f_name)[0]
        hocr_file_path = os.path.join(INPUT_DATA_DIR, f_name)
        
        found_img_path = None
        for ext in image_extensions:
            potential_img_file = os.path.join(INPUT_DATA_DIR, basename + ext)
            if os.path.exists(potential_img_file):
                found_img_path = potential_img_file
                break
        
        if found_img_path:
            basenames_with_paths.append({"basename": basename, "hocr": hocr_file_path, "image": found_img_path})
        else:
            print(f"Warning: HOCR file {f_name} has no corresponding image with extensions {image_extensions}. Skipping.")

print(f"Found {len(basenames_with_paths)} samples with matching image and hOCR files.")
if not basenames_with_paths:
    print("No samples found. Exiting.")
    exit()

random.shuffle(basenames_with_paths)

train_count = int(len(basenames_with_paths) * TRAIN_RATIO)
# The rest will be for validation in this simplified setup
# val_count = len(basenames_with_paths) - train_count 

train_samples = basenames_with_paths[:train_count]
val_samples = basenames_with_paths[train_count:]

datasets = {
    "train": (train_samples, TRAIN_CROP_SUBDIR),
    "val": (val_samples, VAL_CROP_SUBDIR),
}

overall_crop_idx = 0
for set_name, (current_samples, crop_subdir_name) in datasets.items():
    # crop_output_dir is the absolute path to save crops for this set
    crop_output_dir = os.path.join(OUTPUT_RECOGNITION_DATA_DIR, crop_subdir_name) 
    label_file_path = os.path.join(OUTPUT_RECOGNITION_DATA_DIR, f"{set_name}_rec_label.txt")
    
    with open(label_file_path, 'w', encoding='utf-8') as lf:
        
        print(f"Processing {set_name} set ({len(current_samples)} samples)...")
        processed_samples_in_set = 0
        for sample_info in current_samples:
            basename = sample_info["basename"]
            hocr_file = sample_info["hocr"]
            img_file = sample_info["image"]
            
            try:
                image = cv2.imread(img_file)
                if image is None:
                    pil_image = Image.open(img_file).convert('RGB')
                    image = np.array(pil_image)
                    image = image[:, :, ::-1].copy() 
                    if image is None:
                         print(f"Error reading image {img_file} (basename: {basename}). Skipping.")
                         continue
            except Exception as e:
                print(f"Error reading image {img_file} (basename: {basename}): {e}. Skipping.")
                continue

            try:
                with open(hocr_file, 'r', encoding='utf-8') as hf:
                    soup = BeautifulSoup(hf, 'html.parser')
            except Exception as e:
                print(f"Error reading or parsing HOCR file {hocr_file} (basename: {basename}): {e}. Skipping.")
                continue
            
            line_elements = soup.find_all('span', class_=['ocr_line', 'ocr_header'])
            
            for line_idx, line_elem in enumerate(line_elements):
                title = line_elem.get('title', '')
                hocr_bbox = extract_bbox_from_hocr_title(title)
                text_content = line_elem.get_text(separator=' ', strip=True)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

                if hocr_bbox and text_content:
                    cropped_line_img = get_oriented_line_crop(image, hocr_bbox)
                    
                    if cropped_line_img is not None and cropped_line_img.size > 0:
                        crop_filename = f"{basename}_line_{line_idx:03d}_crop_{overall_crop_idx:08d}.png"
                        # relative_crop_path for the label file, relative to OUTPUT_RECOGNITION_DATA_DIR
                        relative_crop_path = os.path.join(crop_subdir_name, crop_filename)
                        abs_crop_save_path = os.path.join(OUTPUT_RECOGNITION_DATA_DIR, relative_crop_path)
                        
                        try:
                            cv2.imwrite(abs_crop_save_path, cropped_line_img)
                            lf.write(f"{relative_crop_path}\t{text_content}\n")
                            all_char_set.update(list(text_content))
                            overall_crop_idx += 1
                        except Exception as e:
                            print(f"Error saving crop {abs_crop_save_path} or writing label: {e}")
                    # else:
                        # print(f"Warning: Crop failed or empty for text '{text_content[:30]}...' in {basename}") # Can be very verbose
            processed_samples_in_set += 1
            if processed_samples_in_set % 100 == 0:
                print(f"  Processed {processed_samples_in_set}/{len(current_samples)} samples in {set_name} set...")


    print(f"Finished {set_name} set. Label file: {label_file_path}")

print(f"\nData preparation complete. {overall_crop_idx} crops generated.")
print(f"Cropped images and label files are in: {OUTPUT_RECOGNITION_DATA_DIR}")

# --- Character Dictionary Generation ---
dict_path = os.path.join(OUTPUT_RECOGNITION_DATA_DIR, "custom_char_dict.txt")
sorted_chars = sorted(list(all_char_set))

with open(dict_path, 'w', encoding='utf-8') as f_dict:
    f_dict.write("<blank>\n") 
    f_dict.write("<unk>\n")   

    if ' ' not in sorted_chars: # Explicitly add space if not found but desired
        print("Adding space ' ' to dictionary as it was not found in extracted text but is often needed.")
        f_dict.write(" \n")

    for char_idx, char_val in enumerate(sorted_chars):
        if char_val == ' ': # If space is already in sorted_chars, ensure it's written correctly
             if ' ' not in all_char_set: # Check if we already wrote it above
                f_dict.write(" \n") # Write space if not already added
        else:
            f_dict.write(f"{char_val}\n")


print(f"\nGenerated a basic character dictionary: {dict_path}")
print("IMPORTANT: Review this dictionary. You might want to merge it with a standard one like")
print("PaddleOCR/ppocr/utils/en_dict.txt (especially for English) and ensure special tokens and space handling are correct.")
print("The <blank> token is typically the first line (index 0) for CTCLoss based models.")