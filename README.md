# Intelligent-Multilingual-Document-Understanding

An interactive web application that allows users to upload documents and ask questions about their content using state-of-the-art AI technology.

## Features

- Upload PDF or image documents
- Interactive question-answering interface
- Support for multiple languages
- Document visualization
- Real-time answers

## Requirements

### Python Dependencies
- Python 3.10+
- See `requirements.txt` for Python package dependencies

### System Dependencies
- Tesseract OCR
- Poppler Utils

## Installation

1. Install system dependencies:

   **Windows:**
   - Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/

   **Linux:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr poppler-utils
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a document and start asking questions!

## Model Training

To train the model on your own data:

1. Ensure you have all dependencies installed
2. Run the training script:
   ```bash
   python scripts/train.py
   ```

The trained model will be saved in the `model_artifacts` directory.

## Project Structure

```
/document-qa
├── app.py              # Main Streamlit application
├── processing.py       # Document processing and OCR
├── model.py           # Model loading and inference
├── scripts/
│   └── train.py       # Model training script
├── model_artifacts/    # Trained model files
├── requirements.txt    # Python dependencies
├── packages.txt       # System dependencies
└── README.md          # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
