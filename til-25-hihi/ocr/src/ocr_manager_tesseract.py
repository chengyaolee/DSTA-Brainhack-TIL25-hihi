import base64
import cv2 # For image decoding
import numpy as np
import os
import subprocess # For running Tesseract
import tempfile
import torch # Though not used for model, to keep device concept if needed elsewhere

class OCRManager:
    """
    Manages OCR inference using Tesseract directly for full image text extraction.
    Aims to be a drop-in replacement for the TrOCR-based OCRManager
    for the predict_batch method's input/output.
    """

    def __init__(self,
                 tesseract_cmd: str = 'tesseract',
                 tesseract_lang: str = 'eng',
                 tesseract_psm: int = 6, # Default PSM: Auto page segmentation with OSD
                 tesseract_oem: int = 1, # Default OCR Engine Mode: Default, based on what is available
                 device: str = None): # Device argument kept for interface consistency if needed
        """
        Initializes the TesseractOCRManager.

        Args:
            tesseract_cmd (str): Command to execute Tesseract. Defaults to 'tesseract'.
                                 Assumes Tesseract is in the system PATH.
            tesseract_lang (str): Language for Tesseract OCR (e.g., 'eng', 'chi_sim').
                                  Ensure the language pack is installed.
            tesseract_psm (int): Tesseract Page Segmentation Mode (0-13).
            tesseract_oem (int): Tesseract OCR Engine Mode (0-3).
            device (str, optional): Kept for interface consistency, not directly used by Tesseract.
                                    If None, autodetects CUDA or defaults to CPU.
        """
        print(f"Initializing TesseractOCRManager with Tesseract command: {tesseract_cmd}")
        self.tesseract_cmd = tesseract_cmd
        self.tesseract_lang = tesseract_lang
        self.tesseract_psm = str(tesseract_psm)
        self.tesseract_oem = str(tesseract_oem)

        if device:
            self.device = device # Stored for consistency, not used by Tesseract directly
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test if Tesseract command is found
        try:
            process = subprocess.run([self.tesseract_cmd, '--version'], capture_output=True, text=True, check=True, timeout=10)
            print(f"Tesseract found: {process.stdout.strip().splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"Error: Tesseract command '{self.tesseract_cmd}' not found or failed to execute.")
            print("Please ensure Tesseract is installed and in your PATH, or provide the full path to 'tesseract_cmd'.")
            print(f"Details: {e}")
            raise RuntimeError(f"Tesseract command '{self.tesseract_cmd}' failed. Is Tesseract installed and in PATH?") from e
        
        print(f"TesseractOCRManager initialized. Language: {self.tesseract_lang}, PSM: {self.tesseract_psm}, OEM: {self.tesseract_oem}")


    def _ocr_single_image_with_tesseract(self, image_file_path: str) -> str:
        """
        Performs OCR on a single image file using Tesseract and returns the extracted text.

        Args:
            image_file_path (str): Path to the input image file.

        Returns:
            str: The extracted text. Returns an empty string if OCR fails.
        """
        command = [
            self.tesseract_cmd,
            image_file_path,
            'stdout', # Output text to stdout
            '-l', self.tesseract_lang,
            '--psm', self.tesseract_psm,
            '--oem', self.tesseract_oem
        ]
        try:
            # print(f"Running Tesseract for direct OCR: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
            # print(f"Tesseract stdout for direct OCR: {process.stdout[:200]}...") # Print a snippet
            # print(f"Tesseract stderr for direct OCR: {process.stderr}")
            return process.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_message = f"Tesseract execution failed with error code {e.returncode} for direct OCR.\n" \
                            f"Command: {' '.join(e.cmd)}\n" \
                            f"Stdout: {e.stdout}\n" \
                            f"Stderr: {e.stderr}"
            print(error_message)
            return "" # Return empty string on error
        except subprocess.TimeoutExpired as e:
            error_message = f"Tesseract execution timed out after {e.timeout} seconds for direct OCR.\n" \
                            f"Command: {' '.join(e.cmd)}"
            print(error_message)
            return ""
        except Exception as e:
            print(f"An unexpected error occurred during Tesseract direct OCR: {e}")
            return ""


    def predict_batch(self, instances: list) -> dict:
        """
        Processes a batch of images provided as base64 encoded strings using Tesseract directly.

        Args:
            instances (list): A list of dictionaries, where each dictionary has a 'b64' key 
                              containing a base64 encoded image string. 
                              Example: [{"key": 0, "b64": "BASE64_IMAGE_DATA"}, ...]

        Returns:
            dict: A dictionary with a "predictions" key, containing a list of
                  transcribed text strings, in the same order as the input instances.
                  Example: {"predictions": ["text1", "text2", ...]}
        """
        predictions = []
        for i, instance_data in enumerate(instances):
            b64_string = instance_data.get("b64")
            # instance_key = instance_data.get("key", f"item_{i}") # Key for logging if needed

            if not b64_string:
                print(f"Warning: Missing 'b64' data for instance {i}. Skipping.")
                predictions.append("") # Append empty string for missing data
                continue

            temp_image_file = None
            try:
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(b64_string)
                    # Convert bytes to OpenCV image format to ensure it's a valid image before saving
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_cv is None:
                        raise ValueError("cv2.imdecode returned None. Invalid image data or format.")
                except Exception as e:
                    print(f"Error decoding base64 string or loading image for instance {i}: {e}")
                    predictions.append("") 
                    continue

                # Create a temporary file for the image (Tesseract needs a file path)
                # Suffix helps Tesseract identify format, e.g., .jpg, .png
                # We save the decoded cv2 image to ensure correct format.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf_img: # Use png or jpg
                    cv2.imwrite(tf_img.name, img_cv)
                    temp_image_file = tf_img.name
                
                # Perform OCR on the temporary image file using Tesseract directly
                transcript = self._ocr_single_image_with_tesseract(temp_image_file)
                predictions.append(transcript)

            except Exception as e: # Catch any other unexpected errors during file handling or OCR
                print(f"Unexpected error processing instance {i}: {e}")
                predictions.append("")
            finally:
                # Clean up temporary image file
                if temp_image_file and os.path.exists(temp_image_file):
                    os.remove(temp_image_file)
        
        return {"predictions": predictions}


# Example Usage (for testing the TesseractOCRManager class directly)
if __name__ == '__main__':
    # --- Configuration for Standalone Test ---
    # This path is for your preprocessed data, used here to get real sample images
    PREPROCESSED_DATA_DIR_FOR_TEST_IMAGES = "./paddleocr_rec_data_prepared2" 
    SAMPLE_IMAGE_SUBDIR_FOR_TEST = "train_crops" # Or "val_crops"
    NUM_SAMPLE_IMAGES_TO_TEST = 2

    # Helper function to read an image file and convert it to a base64 string
    def image_to_base64_string_for_test(file_path: str) -> str | None:
        try:
            with open(file_path, "rb") as image_file:
                img_byte = image_file.read()
            b64_string = base64.b64encode(img_byte).decode('utf-8')
            return b64_string
        except FileNotFoundError:
            print(f"Test Error: Image file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Test Error: Error converting image {file_path} to base64: {e}")
            return None

    print("Standalone Test for TesseractOCRManager:")
    try:
        # Initialize TesseractOCRManager
        # You might need to specify the tesseract_cmd if it's not in your PATH
        # e.g., tesseract_cmd='/usr/bin/tesseract' or 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        # You can also specify language, e.g., tesseract_lang='deu' for German if installed
        ocr_manager = TesseractOCRManager(tesseract_lang='eng') 
        
        # Prepare test instances from actual image files
        sample_images_b64_for_test = []
        images_dir_for_test = os.path.join(PREPROCESSED_DATA_DIR_FOR_TEST_IMAGES, SAMPLE_IMAGE_SUBDIR_FOR_TEST)

        if not os.path.exists(images_dir_for_test) or not os.path.isdir(images_dir_for_test):
            print(f"Test Warning: Sample images directory '{images_dir_for_test}' does not exist. Cannot load real images for test.")
        else:
            print(f"Looking for sample images for test in: {images_dir_for_test}")
            image_files_for_test = [
                f for f in os.listdir(images_dir_for_test)
                if os.path.isfile(os.path.join(images_dir_for_test, f)) and \
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]
            
            if not image_files_for_test:
                print(f"No image files found in {images_dir_for_test} for testing.")
            else:
                for img_file in image_files_for_test[:NUM_SAMPLE_IMAGES_TO_TEST]:
                    img_path = os.path.join(images_dir_for_test, img_file)
                    b64_str = image_to_base64_string_for_test(img_path)
                    if b64_str:
                        sample_images_b64_for_test.append({"path": img_path, "b64": b64_str})
        
        test_instances_for_tesseract_mgr = []
        if sample_images_b64_for_test:
            for i, item in enumerate(sample_images_b64_for_test):
                test_instances_for_tesseract_mgr.append({"key": f"real_img_{i}", "b64": item["b64"], "source_path": item["path"]})
        else:
            print("Test Warning: No real images loaded. Creating a placeholder test instance.")
            # Fallback to a very simple, potentially non-image base64 string if no real images are found
            # Tesseract will likely fail or produce no text, which is a valid test of its behavior.
            # Or, you could integrate the dummy image creation here if PIL is available.
            # For now, just an invalid string to test the flow.
            test_instances_for_tesseract_mgr.append({"key": "placeholder_b64", "b64": "bm90IGFuIGltYWdl", "source_path": "N/A"})


        # Add an invalid base64 case
        test_instances_for_tesseract_mgr.append({"key": "invalid_b64_string", "b64": "this_is_not_base64", "source_path": "N/A"})

        if not test_instances_for_tesseract_mgr:
            print("No test instances could be prepared. Skipping predict_batch test.")
        else:
            print(f"\nRunning predict_batch with {len(test_instances_for_tesseract_mgr)} test instances...")
            results = ocr_manager.predict_batch(test_instances_for_tesseract_mgr)
            
            print("\n--- TesseractOCRManager Batch Prediction Results ---")
            predictions_list = results.get("predictions", [])
            for i, prediction in enumerate(predictions_list):
                instance_key = test_instances_for_tesseract_mgr[i].get('key', i)
                instance_path = test_instances_for_tesseract_mgr[i].get('source_path', 'N/A')
                print(f"Instance Key: {instance_key} (Source: {instance_path})")
                print(f"  Prediction: '{prediction[:200]}{'...' if len(prediction) > 200 else ''}'")
            print("--- End of TesseractOCRManager Batch Prediction Results ---")

    except RuntimeError as e: # Catch initialization error if Tesseract is not found
        print(f"RuntimeError during TesseractOCRManager setup: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during TesseractOCRManager standalone test: {e}")
        import traceback
        traceback.print_exc()