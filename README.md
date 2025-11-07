# heart_bot
🫀 Heartify — Continual Learning Chatbot for Cardiology FAQs
📌 Overview

Heartify is an intelligent, continual learning chatbot designed to assist users with cardiology-related FAQs. It combines extractive and generative AI models to deliver accurate, context-aware answers and adapts over time through experience replay.

The project demonstrates how AI-driven retrieval-augmented generation (RAG) systems can retain domain knowledge while learning from new data — a challenge known as catastrophic forgetting in machine learning.

🎯 Key Features

🧠 Dual-model architecture — combines extractive and generative question-answering models.

🔁 Continual learning — leverages experience replay to maintain knowledge retention (90%+ performance).

🔍 RAG integration — retrieval-augmented generation pipeline for real-time, evidence-backed responses.

⚡ Confidence fallback mechanism — integrates Google Gemini API for handling low-confidence queries.

💬 Interactive API — built for seamless chatbot integration in healthcare platforms.
| Category            | Tools & Frameworks                                 |
| ------------------- | -------------------------------------------------- |
| **Languages**       | Python                                             |
| **AI/ML**           | Transformers (Hugging Face), Scikit-learn, PyTorch |
| **Data Processing** | Pandas, NumPy                                      |
| **Model Serving**   | FastAPI / Flask                                    |
| **Evaluation**      | BLEU, ROUGE, Accuracy, Retention Metrics           |
| **Deployment**      | GitHub + Cloud APIs                                |
