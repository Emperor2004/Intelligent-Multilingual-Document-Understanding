import streamlit as st
from PIL import Image
import pdf2image
import io
import asyncio
from processing import preprocess_document, get_document_image
from model import load_model, get_answer

# Set page config
st.set_page_config(
    page_title="Document Query Engine",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'processed_doc' not in st.session_state:
    st.session_state.processed_doc = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

def main():
    st.title("📄 Document Query Engine")
    st.write("Upload a document and ask questions about its content.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        try:
            # Process the document if not already processed
            if st.session_state.processed_doc is None:
                with st.spinner("Processing document..."):
                    # Get the document image
                    image = get_document_image(uploaded_file)
                    st.session_state.current_image = image
                    # Process the document (OCR)
                    st.session_state.processed_doc = preprocess_document(image)
                st.success("Document processed successfully!")

            # Display the document
            if st.session_state.current_image is not None:
                st.image(st.session_state.current_image, use_column_width=True)

            # Question input
            question = st.text_input(
                "Ask a question about the document",
                placeholder="e.g., What is the effective date?"
            )

            if st.button("Find Answer"):
                if question:
                    try:
                        with st.spinner("Finding answer..."):
                            # Load model and get answer
                            model = load_model()
                            
                            # Process the answer synchronously
                            extracted_answer = get_answer(
                                model,
                                st.session_state.processed_doc,
                                question
                            )
                            
                            # Display the answer
                            st.write("### Answer")
                            st.write(extracted_answer)
                                
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please try again or rephrase your question.")
                else:
                    st.warning("Please enter a question.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
