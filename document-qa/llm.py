import google.generativeai as genai
import streamlit as st
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()
# Configure Gemini
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')  # Make sure to set this environment variable
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_gemini_model():
    """
    Initialize and return the Gemini model.
    """
    return genai.GenerativeModel('gemini-2.5-flash')

def generate_prompt(question: str, context: str, extracted_answer: str = None) -> str:
    """
    Generate a prompt for the LLM that includes the question, context, and extracted answer.
    """
    if extracted_answer:
        return f"""You are an expert document analysis assistant. Based on the following:

Question: {question}

Context from document: {context}

Initial extracted answer: {extracted_answer}

Provide a comprehensive, natural-language answer that:
1. Incorporates the extracted information accurately
2. Provides additional context when relevant
3. Maintains factual accuracy
4. Is clear and well-formatted

Answer: """
    else:
        return f"""You are an expert document analysis assistant. Based on the following:

Question: {question}

Context from document: {context}

Provide a comprehensive, natural-language answer that:
1. Directly answers the question using the provided context
2. Includes relevant details from the context
3. Maintains factual accuracy
4. Is clear and well-formatted

Answer: """

async def get_llm_response(
    question: str,
    context: str,
    extracted_answer: str = None,
    temperature: float = 0.3
) -> str:
    """
    Get a response from Gemini based on the question, context, and extracted answer.
    """
    try:
        model = get_gemini_model()
        prompt = generate_prompt(question, context, extracted_answer)
        
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            ),
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
        )
        
        return response.text
        
    except Exception as e:
        st.error(f"Error in LLM processing: {str(e)}")
        return None
