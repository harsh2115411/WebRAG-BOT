# 🤖 WebRAG Bot - Chat with Web Pages

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that allows you to have interactive conversations with any web page content. Built with **Streamlit**, **LangChain**, and **Groq**.

![WebRAG Bot Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white)

## ✨ Features

- 🌐 **Load any web page** - Simply paste a URL and start chatting
- 💬 **Interactive chat interface** - Beautiful messaging UI with conversation memory
- 🧠 **Smart retrieval** - Uses FAISS vector database for efficient content search
- ⚡ **Fast responses** - Powered by Groq's high-speed LLM inference
- 🎯 **Context-aware** - Maintains conversation history for better responses
- 📱 **Responsive design** - Works on desktop and mobile devices

## 🚀 Live Demo

Try the live app: [WebRAG Bot on Streamlit Cloud](https://your-app-url.streamlit.app)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama-3.3-70B)
- **Embeddings**: OpenAI Embeddings
- **Vector Database**: FAISS
- **Web Scraping**: BeautifulSoup + WebBaseLoader
- **Framework**: LangChain

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.8 or higher
- OpenAI API key
- Groq API key

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/webrag-bot.git
   cd webrag-bot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.streamlit/secrets.toml` file in your project root:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your_openai_api_key_here"
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

## 🏃‍♂️ Running Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How It Works

1. **URL Processing**: The app loads content from the provided web page
2. **Text Chunking**: Content is split into manageable chunks using RecursiveCharacterTextSplitter
3. **Embedding**: Text chunks are converted to vector embeddings using OpenAI
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Query Processing**: User questions are embedded and similar content is retrieved
6. **Response Generation**: Retrieved context is sent to Groq's LLM for answer generation
7. **Memory**: Conversation history is maintained for context-aware responses

```
User Question → Embedding → Vector Search → Context Retrieval → LLM → Response
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your API keys in the Streamlit Cloud secrets management:
   - `OPENAI_API_KEY`
   - `GROQ_API_KEY`
5. Deploy!

### Environment Variables for Deployment

Make sure to set these secrets in your deployment platform:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GROQ_API_KEY`: Your Groq API key

## 📁 Project Structure

```
webrag-bot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml      # API keys (local development)
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## 🎨 Features in Detail

### Smart Document Processing
- Extracts only relevant content using BeautifulSoup
- Efficient text chunking with overlap for context preservation
- Caching for faster subsequent loads

### Advanced Chat Interface
- Modern messaging UI with user/bot message bubbles
- Conversation memory for context-aware responses
- Real-time response streaming
- Mobile-responsive design


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 API Keys

To use this application, you'll need API keys from:

- **OpenAI**: [Get your API key](https://platform.openai.com/api-keys)
- **Groq**: [Get your API key](https://console.groq.com/keys)

## ⚠️ Important Notes

- The app caches processed documents to improve performance
- Large web pages may take a few seconds to process initially
- Make sure your API keys have sufficient credits
- Some websites may block automated scraping

## 🆘 Troubleshooting

### Common Issues

**ModuleNotFoundError**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**API Key Errors**: Ensure your API keys are correctly set in secrets.toml or environment variables

**URL Loading Issues**: Some websites may block scraping. Try different URLs or check if the site allows automated access.

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the [Streamlit documentation](https://docs.streamlit.io)
- Review [LangChain documentation](https://docs.langchain.com)

---

Made with ❤️ by [Your Name]

**Star ⭐ this repository if you found it helpful!**
