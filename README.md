# Text-Summarization

A powerful text summarization tool that uses advanced Natural Language Processing techniques to condense large articles and documents into concise summaries. The application supports multiple input formats including plain text, PDF files, and Wikipedia pages.

## 🎯 Purpose

To save time while reading by summarizing large articles and texts into fewer, meaningful lines. Whether you're dealing with research papers, news articles, or web content, this tool extracts the most important information automatically.

## 📝 Description

This project implements multiple summarization algorithms:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A classical statistical approach for extractive summarization
- **BERT Summarization**: A modern deep learning-based approach using pre-trained language models
- **SpaCy NLP Pipeline**: For advanced natural language processing and text analysis

## ✨ Features

### 📥 Multiple Input Methods
You can summarize text in 4 different ways:

1. **Direct Text Input** - Type or copy-paste text directly into the application
2. **Text File Upload** - Upload and summarize `.txt` files
3. **PDF File Upload** - Extract and summarize PDF documents (with optional page range selection)
4. **Wikipedia URLs** - Automatically scrape and summarize Wikipedia articles by providing the page URL

### 🎛️ Smart Summarization
- Adjustable summary length (percentage-based or fixed ratio)
- Support for both extractive and abstractive summarization
- Automatic text preprocessing and cleaning
- Lemmatization and stopword removal
- BERT-based deep learning summarization for better quality

### 🖥️ Web Interface
- Flask-based web application with intuitive UI
- Responsive design with static CSS styling
- Real-time processing and instant results
- Download summarized content

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Text-Summarization.git
   cd Text-Summarization
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLP models**
   ```bash
   python -c "import spacy; spacy.load('en_core_web_sm')" 
   # If not found, download with:
   python -m spacy download en_core_web_sm
   ```

### Running the Application

**Web Application:**
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

**Command-Line Summarizer:**
```bash
python Text-Summarizer.py
```
Follow the interactive prompts to input text and receive summaries.

## 📦 Project Structure

```
AI-Text-Summarization/
│
├── app.py
├── Text-Summarizer.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── text.ipynb

```

## 📋 Requirements

- **flask** - Web framework for the application
- **spacy** - NLP library with pre-trained models
- **nltk** - Natural Language Toolkit for text processing
- **PyPDF2** - PDF file reading and text extraction
- **beautifulsoup4 (bs4)** - Web scraping for Wikipedia content
- **transformers** - BERT and pre-trained language models
- **numpy** - Numerical computing
- **tensorflow** - Deep learning framework (for BERT)
- **scikit-learn** - Machine learning utilities (TF-IDF vectorization)
- **networkx** - Graph analysis for text summarization

For detailed versions, see `requirements.txt`

## 🔧 Usage Examples

### Web Application
1. Launch the Flask app: `python app.py`
2. Choose input method (text, file, or URL)
3. Adjust summarization parameters (summary percentage)
4. Click "Summarize" to get results
5. View comparison of original vs. summarized text

### Python Script
```python
from Text_Summarizer import bert_summary

text = "Your long article text here..."
summary = bert_summary(text, ratio=0.3)  # 30% summary
print(summary)
```

## 📊 Algorithm Details

### TF-IDF Approach
- Calculates word importance based on frequency in document vs. corpus
- Extracts sentences with highest-scoring words
- Fast and interpretable results

### BERT Approach
- Uses pre-trained transformer models for contextual understanding
- Produces higher-quality summaries for complex texts
- Better for longer documents (50+ words)

## 📈 Performance

- Efficiently summarizes documents of varying lengths
- Processing time depends on text length and algorithm choice
- Typical summarization: <2 seconds for articles under 5000 words

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

Created with ❤️ for NLP and text summarization enthusiasts.

## 🙏 Acknowledgments

- [SpaCy](https://spacy.io/) - Industrial-strength NLP
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit
- [BERT](https://huggingface.co/) - Transformers and pre-trained models
- [Flask](https://flask.palletsprojects.com/) - Web framework

## 📧 Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Happy Summarizing! 📚✨**


## How to install Requirements :-

1. Python3 can be installed from their official site https://www.python.org/ . Or you can use anaconda environment.
2. Spacy can be installed by
For Anaconda Environment > 
```
conda install -c conda-forge spacy

python3 -m spacy download en
```
For other environments > 
```
pip3 install spacy

python3 -m spacy download en
```
3. NLTK can be installed by
For Anaconda Environment > 
```
conda install -c anaconda nltk
```
For other environments > 
```
pip3 install nltk
```

4. PyPdf2 can be installed by
For Anaconda Environment > 
```
conda install -c conda-forge pypdf2
```
For other environments > 
```
pip3 install PyPDF2
```

5. Beautiful Soup (bs4)
For Anaconda Environment > 
```
conda install -c anaconda beautifulsoup4
```
For other environments > 
```
pip3 install beautifulsoup4`
```
## Getting Started :-

- Download or clone repository.

- Open cmd or terminal in same directory where **Text-Summarizer.py** file is stored and then run it by followng command :- 
```
python3 Text-Summarizer.py
```
- Now just follow along with the program.


## Bugs and Improvements :-

- No known bugs. Summary can't be as perfect as humans can do.
- Audio feature will be added soon, so that you can listen the summary too if you want.


## Dev :- Disha M T
