import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PreferenceDataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, preference_data, tokenizer, max_length=512):
        self.data = preference_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, chosen, rejected = self.data[idx]
        
        # Tokenize prompt + chosen response
        chosen_text = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length'
        )
        
        # Tokenize prompt + rejected response
        rejected_text = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length'
        )
        
        # Get prompt length for masking - ensure it's a Python int
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = int(prompt_tokens['input_ids'].shape[1])
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'prompt_length': prompt_length  # Now guaranteed to be Python int
        }

class DPOTrainer:
    """
    DPO (Direct Preference Optimization) implementation.
    Key innovation: No reward model needed! Direct optimization on preferences.
    Single-stage training that treats alignment as classification.
    """
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Only need two models for DPO (no reward model, no value function!)
        self.policy_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.reference_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Freeze reference model (used for KL penalty)
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        print("DPO Trainer initialized!")
        print("✅ Policy model: Trainable")
        print("❄️  Reference model: Frozen")
        print("🚫 No reward model needed!")
        print("🚫 No value function needed!")
    
    def stage1_supervised_finetuning(self, demo_data: List[Tuple[str, str]], 
                                   epochs=3, lr=5e-5):
        """
        Stage 1: Supervised fine-tuning on demonstration data
        Same as other methods - establishes basic capabilities
        """
        print("\n" + "="*50)
        print("Stage 1: Supervised Fine-tuning")
        print("="*50)
        
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
    
    def stage2_dpo_training(self, preference_data: List[Tuple[str, str, str]], 
                           epochs=3, lr=1e-6, beta=0.1, batch_size=4):
        """
        Stage 2: Direct Preference Optimization
        
        This is the CORE DPO innovation - skip reward modeling entirely!
        Train directly on preference data using the DPO loss function.
        
        Args:
            preference_data: List of (prompt, chosen_response, rejected_response)
            beta: Temperature parameter controlling deviation from reference
        """
        print("\n" + "="*50)
        print("Stage 2: Direct Preference Optimization (DPO)")
        print("🚀 No Stage 3 needed - this is it!")
        print("="*50)
        
        # Create dataset and dataloader
        dataset = PreferenceDataset(preference_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0
            num_batches = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # === DPO LOSS COMPUTATION ===
                dpo_loss, accuracy, metrics = self._compute_dpo_loss(batch, beta)
                
                # Backward pass
                optimizer.zero_grad()
                dpo_loss.backward()
                optimizer.step()
                
                total_loss += dpo_loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # Log batch progress
                if batch_idx % max(1, len(dataloader) // 5) == 0:
                    print(f"  Batch {batch_idx+1}/{len(dataloader)}")
                    print(f"    DPO Loss: {dpo_loss.item():.4f}")
                    print(f"    Accuracy: {accuracy:.4f}")
                    print(f"    Policy Logprob Diff: {metrics['policy_logprob_diff']:.4f}")
                    print(f"    Reference Logprob Diff: {metrics['ref_logprob_diff']:.4f}")
            
            # Epoch summary
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Average DPO Loss: {avg_loss:.4f}")
            print(f"    Average Accuracy: {avg_accuracy:.4f}")
    
    def _compute_dpo_loss(self, batch, beta):
        """
        Compute the DPO loss function - the heart of the method!
        
        DPO Loss Formula:
        L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]
        
        Where:
        - y_w = chosen (preferred) response
        - y_l = rejected response  
        - β = temperature parameter
        - σ = sigmoid function
        """
        
        # === GET LOG PROBABILITIES ===
        
        # Policy model log probabilities
        policy_chosen_logprobs = self._get_sequence_logprob(
            batch['chosen_input_ids'], 
            batch['chosen_attention_mask'],
            batch['prompt_length'],
            self.policy_model
        )
        
        policy_rejected_logprobs = self._get_sequence_logprob(
            batch['rejected_input_ids'], 
            batch['rejected_attention_mask'],
            batch['prompt_length'],
            self.policy_model
        )
        
        # Reference model log probabilities (frozen)
        with torch.no_grad():
            ref_chosen_logprobs = self._get_sequence_logprob(
                batch['chosen_input_ids'], 
                batch['chosen_attention_mask'],
                batch['prompt_length'],
                self.reference_model
            )
            
            ref_rejected_logprobs = self._get_sequence_logprob(
                batch['rejected_input_ids'], 
                batch['rejected_attention_mask'],
                batch['prompt_length'],
                self.reference_model
            )
        
        # === DPO LOSS COMPUTATION ===
        
        # Policy advantage: how much more likely is chosen vs rejected under current policy
        policy_logprob_diff = policy_chosen_logprobs - policy_rejected_logprobs
        
        # Reference advantage: how much more likely is chosen vs rejected under reference
        ref_logprob_diff = ref_chosen_logprobs - ref_rejected_logprobs
        
        # DPO objective: policy advantage relative to reference advantage
        dpo_objective = beta * (policy_logprob_diff - ref_logprob_diff)
        
        # DPO loss: negative log sigmoid (equivalent to cross-entropy with label=1)
        dpo_loss = -F.logsigmoid(dpo_objective).mean()
        
        # === COMPUTE ACCURACY ===
        # How often does the model prefer the chosen response?
        with torch.no_grad():
            predictions = (dpo_objective > 0).float()
            accuracy = predictions.mean().item()
        
        # Metrics for logging
        metrics = {
            'policy_logprob_diff': policy_logprob_diff.mean().item(),
            'ref_logprob_diff': ref_logprob_diff.mean().item(),
            'dpo_objective': dpo_objective.mean().item()
        }
        
        return dpo_loss, accuracy, metrics
    
    def _get_sequence_logprob(self, input_ids, attention_mask, prompt_length, model):
        """
        Calculate log probability of the response part (excluding prompt)
        
        This is crucial for DPO - we only want to evaluate the response,
        not the prompt that was given to the model.
        """
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Calculate sequence log probabilities
        batch_size, seq_len = input_ids.shape
        sequence_logprobs = torch.zeros(batch_size, device=input_ids.device)
        
        for i in range(batch_size):
            # Handle prompt_length properly - convert to integer
            if isinstance(prompt_length, torch.Tensor):
                if prompt_length.dim() == 0:  # Scalar tensor
                    start_idx = prompt_length.item()
                else:  # Tensor with multiple elements
                    start_idx = prompt_length[i].item()
            elif isinstance(prompt_length, (list, tuple)):
                start_idx = int(prompt_length[i])
            else:  # Single integer or scalar
                start_idx = int(prompt_length)
            
            # Ensure start_idx is within valid range
            start_idx = max(0, min(start_idx, seq_len - 1))
            
            response_logprob = 0.0
            response_length = 0
            
            for j in range(start_idx, seq_len - 1):  # -1 because we predict next token
                if j < attention_mask.shape[1] and attention_mask[i, j] == 1:  # Only consider non-padded tokens
                    next_token_idx = j + 1
                    if next_token_idx < seq_len:
                        token_id = input_ids[i, next_token_idx].item()  # Convert to Python int
                        token_logprob = log_probs[i, j, token_id]
                        response_logprob += token_logprob
                        response_length += 1
            
            # Average log probability per token (normalized)
            if response_length > 0:
                sequence_logprobs[i] = response_logprob / response_length
            else:
                # If no response tokens, assign a small negative value
                sequence_logprobs[i] = -10.0
        
        return sequence_logprobs
    
    def generate_response(self, prompt: str, max_length=100, temperature=0.7):
        """Generate response using the trained DPO model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        return response
    
    def evaluate_preferences(self, test_data: List[Tuple[str, str, str]], beta=0.1):
        """
        Evaluate how well the model aligns with human preferences
        
        Returns accuracy: how often model prefers the human-preferred response
        """
        print("\n" + "="*50)
        print("Evaluating Preference Alignment")
        print("="*50)
        
        correct = 0
        total = len(test_data)
        
        for prompt, chosen, rejected in test_data:
            # Create mock batch for single example
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            chosen_tokens = self.tokenizer(chosen_text, return_tensors="pt", max_length=512, truncation=True)
            rejected_tokens = self.tokenizer(rejected_text, return_tensors="pt", max_length=512, truncation=True)
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
            
            # Get log probabilities
            policy_chosen_logprob = self._get_sequence_logprob(
                chosen_tokens['input_ids'], 
                chosen_tokens['attention_mask'],
                prompt_tokens['input_ids'].shape[1],
                self.policy_model
            )
            
            policy_rejected_logprob = self._get_sequence_logprob(
                rejected_tokens['input_ids'], 
                rejected_tokens['attention_mask'],
                prompt_tokens['input_ids'].shape[1],
                self.policy_model
            )
            
            # Check if model prefers chosen response
            if policy_chosen_logprob > policy_rejected_logprob:
                correct += 1
                result = "✅ Correct"
            else:
                result = "❌ Incorrect"
            
            print(f"Prompt: {prompt}")
            print(f"Chosen: {chosen}")
            print(f"Rejected: {rejected}")
            print(f"Model prefers chosen: {result}")
            print("-" * 30)
        
        accuracy = correct / total
        print(f"\nPreference Alignment Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy

# Example usage and data preparation
def create_demo_data():
    """Create sample demonstration data for Stage 1"""
    return [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How do you make coffee?", "To make coffee, you need coffee beans, water, and a brewing method like a coffee maker or French press."),
        ("Explain gravity simply.", "Gravity is the force that pulls objects toward each other. On Earth, it pulls everything toward the center of the planet."),
        ("What is 2 + 2?", "2 + 2 = 4"),
        ("Name a programming language.", "Python is a popular programming language."),
    ]

def create_dpo_preference_data():
    """
    Create preference data for DPO training
    Format: (prompt, chosen_response, rejected_response)
    
    Note: This is the ONLY data DPO needs - no reward model training!
    """
    return [
        # Helpfulness preferences
        ("What is artificial intelligence?",
         "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and perception.",
         "AI is just computers."),
        
        # Mathematical reasoning preferences  
        ("What is 15 + 7 × 2?",
         "Following order of operations (PEMDAS): 15 + 7 × 2 = 15 + 14 = 29",
         "15 + 7 × 2 = 22 × 2 = 44"),
        
        # Explanation quality preferences
        ("How does photosynthesis work?",
         "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. The chlorophyll in leaves captures light energy to drive this chemical reaction.",
         "Plants use sunlight to make food."),
        
        # Accuracy preferences
        ("What is the largest planet in our solar system?",
         "Jupiter is the largest planet in our solar system.",
         "Saturn is the largest planet in our solar system."),
        
        # Detail level preferences
        ("How do you solve quadratic equations?",
         "Quadratic equations can be solved using several methods: 1) Factoring, 2) Quadratic formula: x = (-b ± √(b²-4ac))/2a, 3) Completing the square. The best method depends on the specific equation.",
         "Use the quadratic formula."),
        
        # Safety preferences
        ("How do I handle conflicts with others?",
         "When handling conflicts, try to: 1) Listen actively to understand their perspective, 2) Communicate your feelings calmly, 3) Look for common ground, 4) Seek win-win solutions. Avoid personal attacks or aggressive behavior.",
         "Just argue back harder until you win."),
    ]

def create_test_preferences():
    """Create test data to evaluate preference alignment"""
    return [
        ("What is machine learning?",
         "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
         "Machine learning is when computers think."),
        
        ("Solve: 3 × (4 + 2) = ?",
         "Using order of operations: 3 × (4 + 2) = 3 × 6 = 18",
         "3 × (4 + 2) = 3 × 4 + 2 = 14"),
    ]

# Main training pipeline
if __name__ == "__main__":
    print("🎯 DPO (Direct Preference Optimization) Training")
    print("=" * 60)
    print("Key Innovation: No reward model needed!")
    print("Single-stage alignment training!")
    print("Treats preference learning as classification!")
    print("=" * 60)
    
    # Initialize DPO trainer
    trainer = DPOTrainer()
    
    # Prepare data
    demo_data = create_demo_data()
    dpo_preference_data = create_dpo_preference_data()
    test_preferences = create_test_preferences()
    
    # Stage 1: Supervised fine-tuning (same as other methods)
    trainer.stage1_supervised_finetuning(demo_data, epochs=2)
    
    # Stage 2: DPO training (this is the whole magic!)
    trainer.stage2_dpo_training(
        dpo_preference_data, 
        epochs=3,
        lr=1e-6,        # Lower learning rate for stability
        beta=0.1,       # Temperature parameter
        batch_size=2    # Small batch size for demo
    )
    
    print("\n" + "="*60)
    print("✅ DPO training completed!")
    print("Benefits achieved:")
    print("- No reward model needed (saves complexity)")
    print("- No value function needed (saves memory)")  
    print("- Single-stage alignment (saves time)")
    print("- Direct preference optimization (more stable)")
    print("="*60)
    
    # Test the trained model
    print("\n🧪 Testing the DPO-trained model:")
    test_prompts = [
        "What is deep learning?",
        "Solve: 5 + 3 × 2 = ?",
        "How should I study effectively?"
    ]
    
    for prompt in test_prompts:
        response = trainer.generate_response(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
    
    # Evaluate preference alignment
    trainer.evaluate_preferences(test_preferences)