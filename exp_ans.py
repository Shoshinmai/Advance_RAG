from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
import google.generativeai as genai
from dotenv import load_dotenv
import umap

load_dotenv()

gemini_key = os.getenv("Gemini_api_key")
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]
print(word_wrap(pdf_texts[0], width=100))
