import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def invoke(input_text):
    instruction = """You are a top-rated merchandising agent. 
    Be polite to customers and answer all their questions.
    """
    messages = [{"role": "system", "content": instruction},
        {"role": "user", "content": input_text}]
    
    # HuggingFace repository ID
    run_name = "Llama-3.2-Merchandiser"
    repo_id = f"NiazTahi/{run_name}"

    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, legacy=False)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)   
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1200, num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.split("assistant")[1]

    return text

st.title("Merchandiser App")

selected_task = st.sidebar.selectbox("Select NLP Task:", ["Chat with AI"])

input_text = st.text_area("Enter Text:")

if st.button("Process"):
    if selected_task == "Chat with AI" and input_text:
        st.subheader("Generated Text:")
        result = invoke(input_text)
        st.write(result)
    else:
        st.info("Please enter text and select a task from the sidebar.")
