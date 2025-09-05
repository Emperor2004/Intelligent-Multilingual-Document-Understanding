from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForQuestionAnswering,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

def prepare_dataset(examples, processor):
    """
    Prepare dataset examples for training.
    """
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']
    answers = examples['answers']
    
    encoding = processor(
        images,
        words,
        boxes=boxes,
        questions=questions,
        truncation=True,
        padding='max_length'
    )
    
    # Add start and end positions for answers
    start_positions = []
    end_positions = []
    
    for i, answer in enumerate(answers):
        input_ids = encoding.input_ids[i]
        word_ids = encoding.word_ids[i]
        
        # Find start and end of answer in word_ids
        answer_word_start = answer['start']
        answer_word_end = answer['end']
        
        # Convert to token positions
        start_token = None
        end_token = None
        
        for idx, word_id in enumerate(word_ids):
            if word_id == answer_word_start:
                start_token = idx
            if word_id == answer_word_end:
                end_token = idx
                break
        
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = 0
            
        start_positions.append(start_token)
        end_positions.append(end_token)
    
    encoding['start_positions'] = start_positions
    encoding['end_positions'] = end_positions
    
    return encoding

def main():
    # Load DocVQA dataset
    dataset = load_dataset("docvqa")
    
    # Load processor and model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
    
    # Prepare training dataset
    train_dataset = dataset['train'].map(
        lambda examples: prepare_dataset(examples, processor),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./model_artifacts",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained("./model_artifacts/final_model")
    processor.save_pretrained("./model_artifacts/final_model")

if __name__ == "__main__":
    main()
