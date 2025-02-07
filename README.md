An LSTM-powered character-level language model that generates unique and creative company names!

📖 Table of Contents
📊 Dataset
🧠 Architecture
⚡ Optimizer
▶️ How to Run
📊 Dataset
The model is trained on a dataset of 1,146 existing company names, learning their patterns and structures to generate new, similar names.

🧠 Architecture
This project implements an LSTM (Long Short-Term Memory) network from scratch in Python—without using deep learning libraries like TensorFlow or PyTorch.

⚡ Optimizer
The backpropagation algorithm is implemented manually, including gradient computation and parameter updates, making this a true from-scratch deep learning experiment.

▶️ How to Run
Clone the repository and navigate to the project folder:
bash
Copy
Edit
git clone https://github.com/your-username/LSTM-based-name-generator.git  
cd LSTM-based-name-generator
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt  
Set up your Python path:
bash
Copy
Edit
export PYTHONPATH=$PYTHONPATH:$(pwd)
Train the model and generate names:
bash
Copy
Edit
python src/main.py
The number of generated characters can be modified in model.py.
✨ Example Output
After training, the model produces unique, AI-generated company names like:
✅ "Technova"
✅ "Nexora"
✅ "Inovexa"
✅ "Synerflux"

🔥 Want to try it yourself? Run the model and discover brand-new names!

📢 Contributions & Feedback
Have suggestions or ideas? Open an issue or submit a pull request! 🚀
