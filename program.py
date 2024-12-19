import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate

# Specify the local path where the model is stored
local_model_path = "E:/ai-playground/llm/Llama-3.2-3B-Instruct"  # Replace with your local model path

# Load the LLaMA 8B Instruct model and tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Ensure the model is loaded on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a basic prompt template for chat
prompt_template = "You are a helpful assistant. Answer the following question: {question}"
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# Function to generate a response from the LLaMA model
def generate_response(query):
    # Encode the query and prompt
    inputs = tokenizer(prompt.format(question=query), return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(inputs['input_ids'], max_length=200, num_return_sequences=1)
    
    # Decode the response and return it
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
query = "What is the capital of France?"
response = generate_response(query)
print(response)
