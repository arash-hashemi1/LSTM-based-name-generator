# 🚀 LSTM-Based Name Generator  
An **LSTM-powered** character-level language model that generates **unique and creative company names!**  

## 📖 Table of Contents  
- [📊 Dataset](#-dataset)  
- [🧠 Architecture](#-architecture)  
- [⚡ Optimizer](#-optimizer)  
- [▶️ How to Run](#-how-to-run)  

---

## 📊 Dataset  
The model is trained on a dataset of **1,146 existing company names**, learning their patterns and structures to generate new, similar names.  

## 🧠 Architecture  
This project implements an **LSTM (Long Short-Term Memory) network from scratch** in Python—**without using deep learning libraries like TensorFlow or PyTorch.**  

## ⚡ Optimizer  
The **backpropagation algorithm** is implemented manually, including gradient computation and parameter updates, making this a true **from-scratch** deep learning experiment.  

---

## ▶️ How to Run  

1. **Clone the repository** and navigate to the project folder:  
- git clone https://github.com/your-username/LSTM-based-name-generator.git  
- cd LSTM-based-name-generator
- Install the requirements by running `pip install -r requirements.txt`.
- In your terminal, run `export PYTHONPATH=$PYTHONPATH:$(pwd)` to add the current directory to your `PYTHONPATH`.
- Run `python src/main.py` to train the model and generate new company names. The number of generated characters can be modified in the model.py
