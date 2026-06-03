import streamlit as st
import spacy
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Optional NLTK setup
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# -------------------------------
# SAMPLE FAQ DATA (Replace anytime)
# -------------------------------
faqs = {
    "What is your return policy?": "You can return products within 7 days of delivery with original packaging.",
    "How can I track my order?": "You can track your order using the tracking ID sent to your email.",
    "Do you offer customer support?": "Yes, our support team is available 24/7 via chat and email.",
    "How long does delivery take?": "Delivery usually takes 3-5 business days depending on your location.",
    "Can I cancel my order?": "Yes, orders can be cancelled before they are shipped."
    "How can I cancel order?"
    "Can I cancel my purchase?"
    "Cancel my order"
}


questions = list(faqs.keys())
answers = list(faqs.values())

# -------------------------------
# TEXT PREPROCESSING FUNCTION
# -------------------------------
def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.text not in string.punctuation and token.text not in stop_words:
            tokens.append(token.lemma_)
    
    return " ".join(tokens)

# Preprocess FAQ questions
processed_questions = [preprocess(q) for q in questions]

# -------------------------------
# VECTORIZE DATA
# -------------------------------
TfidfVectorizer(ngram_range=(1,2), stop_words="english")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# -------------------------------
# GET BEST RESPONSE
# -------------------------------
def get_response(user_input):
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.15:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing your question."
    
    return answers[index]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 FAQ Chatbot")
st.write("Ask your questions below. The bot will match them with the best FAQ answer.")

# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Type your question here:")

if st.button("Send"):
    if user_input:
        response = get_response(user_input)

        # Save to history
        st.session_state.chat_history.append((user_input, response))

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# -------------------------------
# CHAT HISTORY DISPLAY
# -------------------------------
st.subheader("📜 Chat History")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.markdown("---")

# -------------------------------
# SIDEBAR FEATURES
# -------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("FAQ Bot Features:")

st.sidebar.markdown("""
- NLP Preprocessing (spaCy)
- TF-IDF Matching
- Cosine Similarity
- Chat History
- Fallback Response System
""")

st.sidebar.info("Add your own FAQ dataset in the code to customize this bot.")
import streamlit as st

st.title("FAQ Chatbot")
st.write("If you see this, Streamlit is working correctly.")