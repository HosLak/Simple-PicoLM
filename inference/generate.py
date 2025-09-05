import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blueberry.config import ModelConfig
from blueberry.model import Blueberry


import warnings
warnings.filterwarnings('ignore')

class TextGenerator:
    def __init__(self, model_path: str = "BlueberryModel.pt", device: str = None):
        """Initialize the text generator with a trained model"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Hosseinlack123/Blueberry-testtokenizer")
        
        # Initialize config and model
        self.config = ModelConfig()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Load model
        self.model = Blueberry(self.config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"🔧 Using device: {self.device}")
    
    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 max_length: int = 100,
                 temperature: float = 0.8,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1):
        """
        Generate text based on a prompt
        
        Args:
            prompt: Input text to continue from
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Number of top tokens to consider
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        # Keep track of generated tokens for repetition penalty
        generated_tokens = input_ids[0].tolist()
        
        # Generate tokens
        for _ in range(max_length):
            # Get model predictions
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                logits = self.model(input_ids[:, -self.config.max_seq_len:])
                logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS token is generated (if tokenizer has one)
            if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    def interactive_generate(self):
        """Interactive text generation in terminal"""
        print("\n🤖 Blueberry Text Generator")
        print("Type 'quit' to exit\n")
        
        while True:
            prompt = input("📝 Enter your prompt: ")
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            print("\n🔄 Generating...\n")
            
            generated = self.generate(
                prompt=prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            print("📖 Generated text:")
            print("-" * 50)
            print(generated)
            print("-" * 50 + "\n")

def main():
    # Example usage
    generator = TextGenerator("BlueberryModel.pt")
    
    ## Simple generation
    # prompt = "Once upon a time"
    # result = generator.generate(prompt, max_length=100)
    # print(f"Prompt: {prompt}")
    # print(f"Generated: {result}\n")
    
    # Interactive mode
    generator.interactive_generate()