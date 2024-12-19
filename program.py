import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema import StrOutputParser
import asyncio

async def main():
    # Specify the local path where the model is stored
    local_model_path = "E:/ai-playground/llm/Llama-3.2-3B-Instruct"  # Replace with your local model path

    # Load the LLaMA 8B Instruct model and tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)

    # Ensure the model is loaded on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    hf = HuggingFacePipeline(pipeline=pipe)

    prompt_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    prompt = PromptTemplate(input_variables=["question", "system_message"], template=prompt_template)

    prompt2_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Describe how you would like to remember the quoted text in brief: "{prev}"<|eot_id|>
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    prompt2 = PromptTemplate(input_variables=["prev"], template=prompt2_template)


    l = lambda prev: { "prev" : prev } 
    chain = prompt | hf | StrOutputParser() | l | prompt2  | hf | StrOutputParser()


    async for event in chain.astream_events({"question":input("User: "),"system_message":"you're a helpful assistant"}, version='v2'):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(
                f"{repr(event['data']['chunk'].content)}",
                flush=True, end=""
            )
        if kind == "on_parser_stream":
            print(f"{event['data']['chunk']}", flush=True, end="")

if __name__ == "__main__":
    asyncio.run(main())