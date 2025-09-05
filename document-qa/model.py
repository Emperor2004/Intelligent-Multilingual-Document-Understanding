import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import torch
import numpy as np
from vector_store import VectorStore
from processing import create_text_chunks

@st.cache_resource
def load_model():
    """
    Load the LayoutLMv3 model and processor.
    """
    # Load processor and model from Hugging Face with OCR disabled
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False  # Disable OCR as we're handling it separately
    )
    model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return {
        'processor': processor,
        'model': model
    }

def get_answer(model_dict, doc_data, question):
    """
    Get answer for the question using the model.
    """
    processor = model_dict['processor']
    model = model_dict['model']
    
    try:
        # Ensure words and boxes are properly formatted
        words = doc_data['words']
        boxes = doc_data['boxes']
        
        if not isinstance(words, list):
            words = [str(words)]
            boxes = [boxes[0]] if boxes else [[0, 0, 1000, 1000]]
        
        # Convert all words to strings and ensure they're properly tokenized
        words = [str(word).strip() for word in words if str(word).strip()]
        boxes = boxes[:len(words)]  # Ensure boxes match words
        
        if not words:
            return "No text found in the document."
        
        # Prepare inputs
        try:
            encoding = processor(
                images=doc_data['image'],
                text=question,  # Question as the first sequence
                text_pair=words,  # Document words as a list
                boxes=boxes,  # Matching boxes for each word
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                for key in encoding.keys():
                    if isinstance(encoding[key], torch.Tensor):
                        encoding[key] = encoding[key].cuda()
                model = model.cuda()
            
            # Forward pass
            outputs = model(**encoding)
            
            # Get answer span
            start_logits = outputs.start_logits[0].cpu().detach().numpy()
            end_logits = outputs.end_logits[0].cpu().detach().numpy()
            
            # Get top-k answer spans
            max_answer_length = 50
            n_best_size = 20
            
            start_scores = torch.nn.functional.softmax(torch.tensor(start_logits), dim=-1)
            end_scores = torch.nn.functional.softmax(torch.tensor(end_logits), dim=-1)
            
            # Find the best non-empty answer spans
            best_spans = []
            for start_idx in range(len(start_scores)):
                for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_scores))):
                    score = float(start_scores[start_idx] * end_scores[end_idx])
                    best_spans.append((start_idx, end_idx, score))
            
            # Sort spans by score and take top-k
            best_spans = sorted(best_spans, key=lambda x: x[2], reverse=True)[:n_best_size]
            
            # Get answers from spans
            answers = []
            input_ids = encoding.input_ids[0].cpu().numpy()
            
            for start_idx, end_idx, score in best_spans:
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer_text = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                
                if answer_text and len(answer_text.split()) > 1:  # Ensure answer has at least two words
                    answers.append({
                        'text': answer_text,
                        'score': score
                    })
            
            if not answers:
                return "Could not find an answer in the document."
            
            # Return the best answer with high confidence
            best_answer = answers[0]
            if best_answer['score'] > 0.1:  # Confidence threshold
                return best_answer['text']
            else:
                return "Could not find a confident answer in the document. Please try rephrasing your question."
            
        except Exception as e:
            st.error(f"Error in document processing: {str(e)}")
            return "Could not process the document properly. Please try again."
            
    except Exception as e:
        st.error(f"Error in model inference: {str(e)}")
        return "Could not find an answer. Please try rephrasing your question."
