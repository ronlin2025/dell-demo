## On-premise GPT2 English demo:
[Visit the website](http://103.144.32.3:8080/gpt/demo-gpt2-eng.html)
Demo link: <a href="http://103.144.32.3:8080/gpt/demo-gpt2-eng.html" target="_blank">Click here to visit</a> 
- fine-tune-gpt2-eng.py: This script fine-tunes the lightweight GPT-2 model (distilgpt2) on a custom text dataset (data.txt) using Hugging Face's Transformers on CPU. It tokenizes the data, configures training with the Trainer API, and saves the trained model and tokenizer.
- data.txt: Provide raw text for the GPT-2 model to learn from. Serve as the input for the tokenizer and training process.
- run-gpt2-eng-service.py: This script loads a fine-tuned GPT-2 model and sets up a Flask web service that responds to text prompts via HTTP.
- test-gpt2-eng.py: This script loads a previously fine-tuned GPT-2 model from disk and uses it to generate text based on a given prompt.
## Dell AI RAG demo:
