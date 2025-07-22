# 👗 Fashion AI Agent

An intelligent, multimodal AI-powered assistant for fashion designers and consumers — built using **LangChain**, **OpenAI**, and **Streamlit**.

## 🚀 Features

- 🎨 **Designer Mode**: Helps fashion designers brainstorm creative ideas using text + image input.
- 🛍️ **Customer Mode**: Offers personalized styling recommendations and outfit advice.
- 🔍 **RAG-Enabled Search**: Uses Retrieval-Augmented Generation to provide real-time, relevant suggestions.
- 🖼️ **Image Memory Search**: Allows designers to store and retrieve their favorite styles via vector search.
- ⚡ **Session Memory**: Maintains context-aware conversation for a more natural experience.

## 🧠 Tech Stack

| Component      | Tech Used                              |
| -------------- | -------------------------------------- |
| LLM Backend    | OpenAI GPT-4 (via API)                 |
| Framework      | LangChain                              |
| Frontend       | Streamlit                              |
| Image Search   | FAISS / Sentence Transformers          |
| Image Handling | PIL, OpenCV                            |
| Hosting        | Local / Streamlit Cloud (future ready) |

## 🛠️ Installation

```bash
git clone https://github.com/singla-nitin/Fashion-Ai-Agent.git
cd Fashion-Ai-Agent
pip install -r requirements.txt
streamlit run app/main.py
```
