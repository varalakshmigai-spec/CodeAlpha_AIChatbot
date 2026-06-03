# 🤖 AI FAQ Chatbot

This project is an AI-powered FAQ Chatbot developed using Python and Streamlit. It uses Sentence-BERT and Cosine Similarity to find the most relevant answer to a user's question from a predefined FAQ dataset.

## Features

- FAQ-based question answering
- Semantic matching using Sentence-BERT
- Chat history support
- User-friendly Streamlit interface
- Fast and accurate responses

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- NumPy

## Installation

1. Clone the repository:
2. Install the required packages:
3. Run the application:

## How It Works

1. FAQ questions are stored in a dataset.
2. Sentence-BERT converts questions into embeddings.
3. User questions are compared with FAQ embeddings using cosine similarity.
4. The chatbot returns the most relevant answer.
5. Chat history is displayed during the session.

## Project Structure

```text
ai-faq-chatbot/
│
├── app.py
├── requirements.txt
└── README.md
```

## Future Improvements

- Database integration
- Admin panel for managing FAQs
- Voice-based interaction
- Multi-language support

## Author

Developed as part of an internship project.