---
title: Anime RAG API
emoji: 🌸
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Anime RAG API

This is the FastAPI web server for the Anime RAG application, deployed using Docker on Hugging Face Spaces.

## Local Development

1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Start the FastAPI server locally:
   ```bash
   uvicorn app:app --reload --port 8000
   ```
