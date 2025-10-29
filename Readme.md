# 🧠 Legal Chatbot: AI-Powered Legal Assistant

This project is an AI-based **Legal Chatbot** built using **Python, NLTK, and Deep Learning**.  
It can answer basic legal questions, guide users with information on Indian laws, and simulate legal consultation — helping improve **access to justice** and **legal awareness**.

---

## 🚀 Project Overview

The Legal Chatbot is trained on a custom dataset (`intents.json`) containing categorized intents, user patterns, and responses.  
Using Natural Language Processing (NLP) and a feedforward neural network, it can understand user queries and generate context-based replies.

The project was developed as part of an academic initiative at **VIT Bhopal University**, focusing on the intersection of **AI and Law**.

---

## 🧩 Folder Structure

legal-chatbot/
│
├── data/
│   └── intents.json
│
├── model/
│   └── model.pth
│
├── src/
│   └── chatbotengine.py
│
├── requirements.txt
└── README.md      


---

## ⚙️ Installation and Setup

### 1. Clone this repository

```bash
git clone https://github.com/Ayu-shhh17/legal-chatbot.git
cd legal-chatbot

### 2. Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux

### 3. Install dependencies

pip install -r requirements.txt

## 🧠 Training the Model

Run the following command to train and save the model:

python -m src.chatbotengine


Once trained, the model and tokenizer will be automatically saved in the model/ folder.

💬 Running the Chatbot

After training, start the chatbot using:

python chatbot.py


The chatbot will launch in your terminal.
You can then start asking legal questions such as:

> What is an FIR?
> How can I file a complaint?
> What are my rights as a consumer?

🧾 Example Interaction
User: What is bail?
Bot: Bail is the conditional release of a defendant with the promise to appear in court when required.

🧠 Tech Stack

Programming Language: Python

Libraries Used: TensorFlow / Keras, NLTK, NumPy, Pickle

Architecture: Feedforward Neural Network (Sequential Model)

Dataset: Custom JSON-based intents file

🌐 Applications

🧾 Legal Consultation & Advice – Helps users understand legal terms and procedures.

⚖️ Access to Justice – Provides basic legal guidance for those who cannot afford lawyers.

🧮 Document Assistance – Can be extended to draft legal templates or forms.

🔍 Case Research Aid – Can fetch references or past case laws (in advanced versions).

📈 Future Improvements

Integrate Dialogflow or Rasa for context-based conversations

Add speech-to-text and text-to-speech support

Deploy as a web-based legal assistant

Expand dataset with regional laws and multilingual support

🧑‍💻 Author

Ayush Philip George
AI & ML Enthusiast | CSE (AI) – VIT Bhopal University
📧 ayushgeorge710@gmail.com

🌐 GitHub: Ayu-shhh17

🪪 License

This project is released under the MIT License.
Feel free to use, modify, and improve it for educational or research purposes.

⭐ If you found this project helpful, consider giving it a star on GitHub!
