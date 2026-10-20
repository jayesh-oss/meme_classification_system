
# 🧠 Meme Classification System  
### Machine Learning–Based Image Classification Web Application

---

## 📌 Overview

The **Meme Classification System** is an end-to-end **machine learning–based web application** that classifies meme images into predefined categories.  
The project demonstrates the practical application of **machine learning, image processing, and web deployment** using Python and Flask.

It covers the complete ML workflow including **image preprocessing, model inference, and result visualization** through a web interface.

---

## 🎯 Motivation

With the rapid growth of social media, memes have become a major form of digital communication. Manual categorization of memes is inefficient and time-consuming.  
This project aims to provide an **automated and scalable solution** for meme classification using machine learning techniques.

---

## 🧠 System Workflow

1. User uploads a meme image through the web interface  
2. Image is preprocessed for model compatibility  
3. CLIP compares the image against five category prompts (or a trained fusion head runs)  
4. Predicted meme category and confidence scores are displayed to the user  

---

## 🚀 Key Features

- Image-based meme classification  
- Web application built using Flask  
- Machine learning–driven prediction system  
- Modular and organized project structure  
- Easy to extend with advanced models  

---

## 🛠️ Technologies Used

### Programming & Frameworks
- Python  
- Flask  

### Machine Learning & Image Processing
- PyTorch  
- Hugging Face Transformers  
- CLIP (ViT-B/32) zero-shot classification  
- sentence-transformers (OCR text embeddings)  
- PyTesseract OCR  
- NumPy / Pillow / scikit-learn  

### Frontend
- HTML5  
- CSS3  

---

## 📂 Project Structure

```
meme_classification_system/
│
├── app.py                 # Flask web application
├── history.py             # Prediction history / logging
├── utils/                 # Utility functions
├── models/                # Trained ML model files
├── static/                # Static assets (CSS, images)
├── templates/             # HTML templates
├── requirements.txt       # Project dependencies
└── sample.jpg          # Sample meme image
```

---

## 🧠 Model Pipeline

The default classifier is CLIP `openai/clip-vit-base-patch32` used as zero-shot image classification against five prompts. It needs no training data and works on CPU or GPU.

For higher accuracy, the project also supports a fusion head that combines:
- CLIP image embeddings (512-d)
- MiniLM OCR text embeddings (384-d)
- A small MLP trained on labeled memes

If `models/meme_head.pt` exists, inference uses it automatically and falls back to CLIP zero-shot otherwise.

### Train the fusion head

Organize labeled images into class folders:

```
data/memes/
├── Wholesome/
├── Dark/
├── Political/
├── Promotional/
└── Offensive/
```

Then precompute embeddings and train the head:

```bash
python training/prepare_embeddings.py --data-dir data/memes
python training/train_head.py --embeddings data/embeddings.npz
```

The training script reports macro-F1 and a confusion matrix, then saves `models/meme_head.pt`.

### GPU note (Windows)

The default `pip install torch` wheel is CPU-only. For an NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

---

## ⚙️ Installation & Execution

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jayesh-oss/meme_classification_system.git
cd meme_classification_system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app.py
```

> OCR requires a Tesseract installation. Set `TESSERACT_CMD` if it is not on your PATH.
> Set `CLIP_MODEL` to override the default model, or `FUSION_HEAD` to point at a custom fusion checkpoint.

Open in browser:
```
http://127.0.0.1:5000/
```

---

## 📈 Use Cases

- Automated meme categorization  
- Social media content analysis  
- Image classification learning project  
- Academic machine learning application  

---

## 🔮 Future Enhancements

- Integration of deep learning (CNN) models  
- Support for additional meme categories  
- Display prediction confidence score  
- Cloud deployment (Render / AWS)  
- Improved user interface  

---

## ⚠️ Disclaimer

This project is developed **strictly for educational purposes** and is not intended for commercial content moderation.

---

## 👨‍💻 Author

**Jayesh Wable**  
B.Tech – Computer Science & Engineering (AI & ML)  
GitHub: https://github.com/jayesh-oss
