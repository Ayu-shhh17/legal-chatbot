# 🏛️ Legal Chatbot - AI Solutions for Modern Justice

This project is a smart AI-powered **Legal Chatbot** designed to provide legal assistance, document generation, case research support, and access to justice in an easy and automated way. It aims to bridge the gap between people and the legal system using modern AI techniques.

## 📂 Project Structure

- `legal_chatbot.py` → Python code for the chatbot engine.
- `intents.json` → Custom-built intents and response dataset.
- `nltk_utils.py` → Preprocessing tools (tokenization, stemming).
- `model.pth` → Trained machine learning model for intent classification.
- `README.md` → Documentation of the project.

## 🛠️ Technologies Used

- Python
- Natural Language Toolkit (NLTK)
- PyTorch (for building and training the model)
- JSON for intents management
- Flask (optional for web deployment)

## 🧠 Features

- 📄 Legal Document Generation and Review
- 🧑‍⚖️ Legal Consultation and Advice (basic level)
- 📚 Legal Research Assistance (case and topic searching)
- ⚖️ Court and Case Management Help
- 🤝 Access to Justice & Pro Bono Services Support

## 📊 Workflow

1. User inputs a question or query.
2. The chatbot preprocesses the input (tokenization + stemming).
3. The trained model predicts the intent.
4. A response is selected based on the intent detected.
5. Chatbot replies to the user.

## 🚀 How to Run

1. Clone the repository:

https://github.com/Ayu-shhh17/legal-chatbot.git

2. Install the required packages:

pip install nltk
            torch
            flask

3. Run the chatbot:

legal_chatbot.py


(If using Flask for web hosting, run the `app.py` separately.)

## 🔥 Example Intents

- "What is intellectual property law?"
- "Tell me about family law"
- "What are my tenant rights?"
- "What is the procedure for filing a case?"

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

### ✨ Future Improvements

- Integrate with live legal databases.
- Add multilingual support.
- Enhance natural language understanding with transformers (BERT/LLMs).
- Build a complete web or mobile app frontend.

---

### 👨‍💻 Author

- [Ayush George](https://github.com/Ayu-shhh17)




