# CorpBrain: The Autonomous Meeting Intelligence Platform

CorpBrain is an AI system that automatically ingests meeting recordings, understands them, builds a "Knowledge Graph" of the company's decisions, and executes tasks based on what was said.

## Architecture

1.  **Ingestion Service**: Handles file uploads and audio processing.
2.  **Cognitive Service**: Runs AI models (Transcription + NLP).
3.  **Knowledge Service**: Stores vector embeddings for RAG.
4.  **Agentic Service**: Executes tasks based on insights.
