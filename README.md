# OCR Data Extraction Project

## Task Description
The task involves extracting data via OCR (Optical Character Recognition) from an input image. The goal is to recognize and extract texts from the image, particularly the names of cities such as Genoa and Camaldoli, along with their bounding boxes and confidence scores. The final output is provided in JSON format.

This project provides a method to interact with a satellite image (in PNG format) and extract relevant information from it.

## Project Structure
- **OCR Tool:** PaddleOCR
- **LLM Model:** Zephyr-7B
- **Input:** Image (PNG format)
- **Output:** JSON format containing extracted text, bounding boxes, and confidence scores.

## Prerequisites
- Python 3.8 or later
- `git`
- `pip`

## Installation
1. **Clone the PaddleOCR Repository:**
   ```sh
   git clone https://github.com/PaddlePaddle/PaddleOCR.git
   ```

2. **Install Dependencies:**
   ```sh
   pip install paddlepaddle-gpu
   pip install paddleocr
   ```

3. **Install Zephyr-7B Dependencies:**
   ```sh
   pip install git+https://github.com/huggingface/transformers.git
   pip install accelerate
   ```

## Setup and Usage
1. **Clone the PaddleOCR Repository:**
   ```sh
   !git clone https://github.com/PaddlePaddle/PaddleOCR.git
   ```

2. **Install PaddleOCR and its dependencies:**
   ```sh
   !pip install paddlepaddle-gpu
   !pip install paddleocr
   ```

3. **Load and visualize the input image:**
   ```python
   from google.colab.patches import cv2_imshow
   import cv2

   img_path = '/content/Genova.png'
   img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
   cv2_imshow(img)
   ```

4. **Perform OCR using PaddleOCR:**
   ```python
   from paddleocr import PaddleOCR

   ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)
   result = ocr.ocr(img_path, cls=True)
   ```

5. **Convert OCR result to Pandas DataFrame for better visualization:**
   ```python
   import pandas as pd

   flattened_data = []
   for sublist in result:
       for item in sublist:
           flattened_data.append({'coordinates': item[0], 'label': item[1][0], 'confidence': item[1][1]})

   df = pd.DataFrame(flattened_data)
   df.head()
   ```

6. **Extract OCR results as a string for further processing:**
   ```python
   ocr_string = ""
   for i in range(len(result[0])):
       ocr_string += result[0][i][1][0] + " "
   ```

7. **Parse OCR output using Zephyr-7B LLM:**
   ```python
   import torch
   from transformers import pipeline

   pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

   messages = [
       {
           "role": "system",
           "content": "You are a JSON converter which receives raw boarding pass OCR information as a string and returns a structured JSON output by organising the information in the string.",
       },
       {"role": "user", "content": f"Extract the name of all the Cities this OCR data: {ocr_string}"},
   ]

   prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   outputs = pipe(prompt, max_new_tokens=250, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

   print(outputs[0]["generated_text"])
   ```

## Example JSON Output
```json
{
  "Cities": [
    "Camaldoli",
    "Genes",
    "Tirreni",
    "Porto"
  ]
}
```

## Explanation
1. The JSON object contains a key `Cities` which holds an array of city names extracted from the given OCR data.
2. The OCR data is processed to identify substrings that match known city names.
3. The identified city names are listed in the JSON array for structured output.

## Notes
- Ensure that the image file (e.g., Genova.png) is available in the working directory.
- The given instructions include steps to install and configure dependencies for both PaddleOCR and Zephyr-7B.

## Requirements
List of required Python packages:
```txt
paddlepaddle-gpu
paddleocr
git+https://github.com/huggingface/transformers.git
accelerate
```
