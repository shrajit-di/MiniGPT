"""
ui/app.py
=========
Flask API backend that connects our PyTorch MiniGPT model to the web frontend.
"""
import os
import sys
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add the parent directory and src to sys.path so we can import our model code seamlessly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from model import MiniGPT
from tokenizer import CharTokenizer

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    global model, tokenizer
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "minigpt_final.pt")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input.txt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return False
        
    # Load dataset to get exact tokenizer mappings
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.vocab_size = checkpoint['vocab_size']
    
    # Initialize model
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded. Train the model first!"}), 500
        
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 100))
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 5))
    
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # Encode prompt
    context = tokenizer.encode(prompt)
    x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    generated_text = tokenizer.decode(y[0].tolist())
    
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    if load_model():
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to start server. Please train the model first.")
