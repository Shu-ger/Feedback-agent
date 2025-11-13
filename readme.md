# Feedback Agent

A lightweight FastAPI-based service that analyzes user feedback via a local model.

Please download mistral-7b-openorca.Q4_0.gguf from 
https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q4_0.gguf

### 🚀 Build and Run

```bash
docker build -t feedback-agent .
docker run -p 8000:8000 feedback-agent
