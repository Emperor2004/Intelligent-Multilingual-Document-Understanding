import streamlit as st
from PIL import Image
import pytesseract
import pdf2image
import numpy as np
import io
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from vector_store import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Poppler path
POPPLER_PATH = r'C:\poppler\Library\bin'
if not Path(POPPLER_PATH).exists():
    st.error(f"Poppler not found at {POPPLER_PATH}. Please ensure it's installed correctly.")
    if not any(Path(p).name == 'pdftoppm.exe' for p in os.environ['PATH'].split(os.pathsep)):
        st.error("Poppler is not found in system PATH. Please install Poppler and add it to PATH.")

@st.cache_data
def create_text_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_data
def preprocess_document(_image):
    """
    Process the document image using Tesseract OCR.
    Returns the OCR results including words and their bounding boxes.
    The leading underscore in _image prevents Streamlit from trying to hash the PIL Image.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(_image, Image.Image):
        image_np = np.array(_image)
    else:
        image_np = _image

    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    
    # Filter out empty text entries and prepare word-box pairs
    words = []
    boxes = []
    
    img_h, img_w = image_np.shape[:2]
    
    # Process OCR data to extract words and their positions
    words = []
    boxes = []
    word_positions = []  # To maintain word order
    line_texts = {}  # Store text by line number
    
    # First pass: Group text by lines
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            y = ocr_data['top'][i]
            h = ocr_data['height'][i]
            line_num = int(y / (h * 0.8))
            
            if line_num not in line_texts:
                line_texts[line_num] = []
            line_texts[line_num].append({
                'text': text,
                'x': ocr_data['left'][i],
                'y': y,
                'w': ocr_data['width'][i],
                'h': h
            })
    
    # Second pass: Process each line
    for line_num in sorted(line_texts.keys()):
        # Sort words in line by x position
        line_words = sorted(line_texts[line_num], key=lambda x: x['x'])
        
        for word_info in line_words:
            text = word_info['text'].strip()
            if text:
                x = word_info['x']
                y = word_info['y']
                w = word_info['w']
                h = word_info['h']
                
                # Split text into individual words
                word_list = text.split()
                word_width = w / len(word_list) if len(word_list) > 0 else w
            
            for idx, word in enumerate(word_list):
                if word.strip():
                    word_x = x + (idx * word_width)
                    
                    # Normalize coordinates to [0, 1000] range
                    x1 = min(max(int(word_x * 1000 / img_w), 0), 1000)
                    y1 = min(max(int(y * 1000 / img_h), 0), 1000)
                    x2 = min(max(int((word_x + word_width) * 1000 / img_w), 0), 1000)
                    y2 = min(max(int((y + h) * 1000 / img_h), 0), 1000)
                    
                    words.append(word.strip())
                    boxes.append([x1, y1, x2, y2])
                    word_positions.append((line_num, x1))  # Store position for sorting
                    
    # Sort words by line number and horizontal position
    sorted_indices = sorted(range(len(word_positions)), key=lambda k: (word_positions[k][0], word_positions[k][1]))
    words = [words[i] for i in sorted_indices]
    boxes = [boxes[i] for i in sorted_indices]
    
    # Return preprocessed document data
    document_text = ' '.join(words)
    
    return {
        'words': words,
        'boxes': boxes,
        'image': _image
    }

@st.cache_data
def get_document_image(uploaded_file):
    """
    Convert the uploaded file to a PIL Image.
    Handles both PDF and image files.
    """
    # Get file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        try:
            # Convert PDF to image (first page only)
            pdf_bytes = uploaded_file.read()
            # Explicitly set poppler path and add error handling
            images = pdf2image.convert_from_bytes(
                pdf_bytes,
                poppler_path=POPPLER_PATH,
                first_page=1,
                last_page=1
            )
            return images[0]  # Return first page
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.error("Please ensure Poppler is installed correctly at: " + POPPLER_PATH)
            return None
    else:
        # For image files
        return Image.open(uploaded_file)
