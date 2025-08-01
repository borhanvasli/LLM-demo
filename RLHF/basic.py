import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class RLHFTrainer:
    """
    Simplified RLHF implementation for educational purposes.
    In practice, you'd use more sophisticated implementations like trlX or DeepSpeed-Chat.
    """
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Stage 1: Base model (pretrained)
        self.policy_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.reference_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Stage 2: Reward model
        self.reward_model = RewardModel(self.policy_model.config.hidden_size)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
    
    def stage1_supervised_finetuning(self, demo_data: List[Tuple[str, str]], 
                                   epochs=3, lr=5e-5):
        """
        Stage 1: Supervised fine-tuning on demonstration data
        demo_data: List of (prompt, desired_response) pairs
        """
        print("Stage 1: Supervised Fine-tuning...")
        
        optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for prompt, response in demo_data:
                # Tokenize input
                full_text = prompt + response
                inputs = self.tokenizer(full_text, return_tensors="pt", 
                                      truncation=True, max_length=512)
                
                # Forward pass
                outputs = self.policy_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(demo_data):.4f}")
    
    def stage2_reward_model_training(self, preference_data: List[Tuple[str, str, str, int]], 
                                   epochs=3, lr=1e-4):
        """
        Stage 2: Train reward model on human preferences
        preference_data: List of (prompt, response1, response2, preference) 
        where preference is 0 if response1 is preferred, 1 if response2 is preferred
        """
        print("\nStage 2: Reward Model Training...")
        
        optimizer = optim.AdamW(self.reward_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for prompt, resp1, resp2, preference in preference_data:
                # Get embeddings for both responses
                emb1 = self._get_response_embedding(prompt + resp1)
                emb2 = self._get_response_embedding(prompt + resp2)
                
                # Get reward scores
                reward1 = self.reward_model(emb1.unsqueeze(0))
                reward2 = self.reward_model(emb2.unsqueeze(0))
                
                # Create preference logits
                logits = torch.cat([reward1, reward2], dim=1)
                target = torch.tensor([preference], dtype=torch.long)
                
                # Calculate loss
                loss = criterion(logits, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
            
            accuracy = correct / len(preference_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(preference_data):.4f}, "
                  f"Accuracy: {accuracy:.4f}")
    
    def stage3_ppo_training(self, prompts: List[str], epochs=3, lr=1e-5, 
                           kl_coeff=0.1, clip_epsilon=0.2):
        """
        Stage 3: PPO training using reward model
        This is a simplified version - real PPO is more complex
        """
        print("\nStage 3: PPO Training...")
        
        optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_reward = 0
            total_kl = 0
            
            for prompt in prompts:
                # Generate response with current policy
                response = self._generate_response(prompt)
                
                # Calculate reward
                embedding = self._get_response_embedding(prompt + response)
                reward = self.reward_model(embedding.unsqueeze(0))
                
                # Calculate KL divergence penalty
                kl_penalty = self._calculate_kl_penalty(prompt + response)
                
                # Total objective (reward - KL penalty)
                objective = reward - kl_coeff * kl_penalty
                
                # Simple policy gradient update (simplified PPO)
                loss = -objective
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reward += reward.item()
                total_kl += kl_penalty.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Reward: {total_reward/len(prompts):.4f}, "
                  f"Avg KL: {total_kl/len(prompts):.4f}")
    
    def _get_response_embedding(self, text: str):
        """Get embedding representation of text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.policy_model.transformer(**inputs)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding
    
    def _generate_response(self, prompt: str, max_length=100):
        """Generate response using current policy"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], 
                                       skip_special_tokens=True)
        return response
    
    def _calculate_kl_penalty(self, text: str):
        """Calculate KL divergence between current policy and reference model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get logits from both models
        policy_outputs = self.policy_model(**inputs)
        with torch.no_grad():
            ref_outputs = self.reference_model(**inputs)
        
        # Calculate KL divergence
        policy_logprobs = torch.log_softmax(policy_outputs.logits, dim=-1)
        ref_logprobs = torch.log_softmax(ref_outputs.logits, dim=-1)
        
        kl_div = torch.sum(torch.exp(ref_logprobs) * (ref_logprobs - policy_logprobs))
        return kl_div

class RewardModel(nn.Module):
    """Simple reward model that maps embeddings to scalar rewards"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, embedding):
        return self.layers(embedding)

# Example usage and data preparation
def create_demo_data():
    """Create sample demonstration data for Stage 1"""
    return [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How do you make coffee?", "To make coffee, you need coffee beans, water, and a brewing method like a coffee maker or French press."),
        ("Explain gravity simply.", "Gravity is the force that pulls objects toward each other. On Earth, it pulls everything toward the center of the planet."),
    ]

def create_preference_data():
    """Create sample preference data for Stage 2"""
    return [
        ("What is AI?", 
         "AI is artificial intelligence.", 
         "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.", 
         1),  # Second response preferred
        ("How does learning work?", 
         "Learning happens in the brain.", 
         "Learning is the process by which we acquire new knowledge, skills, or behaviors through experience, study, or instruction.", 
         1),  # Second response preferred
    ]

def create_prompts():
    """Create sample prompts for Stage 3"""
    return [
        "Explain quantum physics in simple terms:",
        "What are the benefits of exercise?",
        "How do computers work?",
    ]

# Main training pipeline
if __name__ == "__main__":
    # Initialize trainer
    trainer = RLHFTrainer()
    
    # Prepare data
    demo_data = create_demo_data()
    preference_data = create_preference_data()
    prompts = create_prompts()
    
    # Stage 1: Supervised fine-tuning
    trainer.stage1_supervised_finetuning(demo_data, epochs=2)
    
    # Stage 2: Reward model training
    trainer.stage2_reward_model_training(preference_data, epochs=2)
    
    # Stage 3: PPO training
    trainer.stage3_ppo_training(prompts, epochs=2)
    
    print("\nRLHF training completed!")
    
    # Test the trained model
    test_prompt = "What is machine learning?"
    response = trainer._generate_response(test_prompt)
    print(f"\nTest generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")