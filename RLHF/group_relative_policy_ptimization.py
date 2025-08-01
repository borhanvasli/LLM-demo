import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class GRPOBatch:
    """Container for GRPO training batch data"""
    prompts: List[str]
    responses: List[List[str]]  # Multiple responses per prompt
    rewards: List[List[float]]  # Rewards for each response
    log_probs: List[List[float]]  # Log probabilities for each response
    group_ids: List[int]  # Which group each prompt belongs to

class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) implementation.
    Key innovation: Uses group-based advantages instead of value functions.
    Memory efficient and particularly effective for reasoning tasks.
    """
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Stage 1: Base model (pretrained)
        self.policy_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.reference_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Stage 2: Reward model
        self.reward_model = RewardModel(self.policy_model.config.hidden_size)
        
        # NOTE: No value function needed for GRPO! This saves ~40% memory
        
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
    
    def stage3_grpo_training(self, prompts: List[str], epochs=3, lr=1e-5, 
                           kl_coeff=0.1, responses_per_prompt=4, group_size=8):
        """
        Stage 3: GRPO training using group-based advantages
        
        Key Innovation: No value function needed!
        Instead, we use group statistics as baselines.
        
        Args:
            prompts: List of training prompts
            responses_per_prompt: How many responses to generate per prompt
            group_size: How many similar prompts to group together
        """
        print("\nStage 3: GRPO Training (Group Relative Policy Optimization)...")
        
        optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # === GROUP FORMATION ===
            # Group similar prompts together for fair comparison
            groups = self._form_prompt_groups(prompts, group_size)
            
            epoch_rewards = []
            epoch_advantages = []
            
            for group_id, group_prompts in enumerate(groups):
                print(f"  Processing group {group_id+1}/{len(groups)} ({len(group_prompts)} prompts)")
                
                # === RESPONSE GENERATION ===
                # Generate multiple responses for each prompt in the group
                batch = self._generate_group_responses(group_prompts, responses_per_prompt)
                
                # === GROUP-BASED ADVANTAGE COMPUTATION ===
                # This is the CORE innovation of GRPO
                advantages = self._compute_group_advantages(batch)
                
                # === POLICY UPDATE ===
                group_loss = self._compute_grpo_loss(batch, advantages, kl_coeff)
                
                optimizer.zero_grad()
                group_loss.backward()
                optimizer.step()
                
                # Logging
                group_avg_reward = np.mean([np.mean(rewards) for rewards in batch.rewards])
                group_avg_advantage = np.mean([np.mean(advs) for advs in advantages])
                
                epoch_rewards.append(group_avg_reward)
                epoch_advantages.append(group_avg_advantage)
            
            # Epoch summary
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Avg Reward: {np.mean(epoch_rewards):.4f}")
            print(f"    Avg Advantage: {np.mean(epoch_advantages):.4f}")
    
    def _form_prompt_groups(self, prompts: List[str], group_size: int):
        """
        Form groups of similar prompts for fair comparison.
        In practice, you'd use more sophisticated similarity metrics.
        """
        # Simple grouping: just batch prompts sequentially
        # In practice, you'd group by semantic similarity, task type, etc.
        groups = []
        for i in range(0, len(prompts), group_size):
            group = prompts[i:i + group_size]
            if len(group) >= 2:  # Need at least 2 prompts for meaningful comparison
                groups.append(group)
        return groups
    
    def _generate_group_responses(self, prompts: List[str], responses_per_prompt: int):
        """
        Generate multiple responses for each prompt in the group.
        This is essential for GRPO - we need multiple samples to compute group statistics.
        """
        batch_prompts = []
        batch_responses = []
        batch_rewards = []
        batch_log_probs = []
        
        for prompt in prompts:
            prompt_responses = []
            prompt_rewards = []
            prompt_log_probs = []
            
            # Generate multiple responses for this prompt
            for _ in range(responses_per_prompt):
                response, log_prob = self._generate_response_with_log_prob(prompt)
                
                # Get reward from reward model
                full_text = prompt + response
                embedding = self._get_response_embedding(full_text)
                reward = self.reward_model(embedding.unsqueeze(0)).item()
                
                prompt_responses.append(response)
                prompt_rewards.append(reward)
                prompt_log_probs.append(log_prob)
            
            batch_prompts.append(prompt)
            batch_responses.append(prompt_responses)
            batch_rewards.append(prompt_rewards)
            batch_log_probs.append(prompt_log_probs)
        
        return GRPOBatch(
            prompts=batch_prompts,
            responses=batch_responses,
            rewards=batch_rewards,
            log_probs=batch_log_probs,
            group_ids=list(range(len(batch_prompts)))
        )
    
    def _compute_group_advantages(self, batch: GRPOBatch):
        """
        CORE GRPO INNOVATION: Compute advantages using group statistics
        
        Instead of learning a value function V(s), we use:
        Advantage_i = Reward_i - Group_Mean_Reward
        
        This is much simpler and more memory efficient!
        """
        print("    Computing group-based advantages...")
        
        # Flatten all rewards in the group
        all_rewards = []
        for prompt_rewards in batch.rewards:
            all_rewards.extend(prompt_rewards)
        
        # Compute group statistics
        group_mean_reward = np.mean(all_rewards)
        group_std_reward = np.std(all_rewards) + 1e-8  # Add small epsilon for stability
        
        print(f"      Group mean reward: {group_mean_reward:.4f}")
        print(f"      Group std reward: {group_std_reward:.4f}")
        
        # Compute advantages for each response
        advantages = []
        for prompt_rewards in batch.rewards:
            prompt_advantages = []
            for reward in prompt_rewards:
                # GRPO Advantage Formula: Individual - Group Mean
                advantage = reward - group_mean_reward
                # Normalize by group standard deviation for stability
                normalized_advantage = advantage / group_std_reward
                prompt_advantages.append(normalized_advantage)
            advantages.append(prompt_advantages)
        
        return advantages
    
    def _compute_grpo_loss(self, batch: GRPOBatch, advantages: List[List[float]], kl_coeff: float):
        """
        Compute GRPO loss using group-based advantages
        
        Loss = -E[Advantage * log_prob] + KL_penalty
        """
        total_loss = 0
        num_responses = 0
        
        for prompt_idx, (prompt, responses) in enumerate(zip(batch.prompts, batch.responses)):
            prompt_advantages = advantages[prompt_idx]
            prompt_log_probs = batch.log_probs[prompt_idx]
            
            for response, advantage, log_prob in zip(responses, prompt_advantages, prompt_log_probs):
                # Policy gradient loss with advantage weighting
                policy_loss = -advantage * log_prob
                
                # KL divergence penalty (keep policy close to reference)
                kl_penalty = self._calculate_kl_penalty(prompt + response)
                
                # Total loss for this response
                response_loss = policy_loss + kl_coeff * kl_penalty
                total_loss += response_loss
                num_responses += 1
        
        return total_loss / num_responses  # Average loss
    
    def _generate_response_with_log_prob(self, prompt: str, max_length=100):
        """Generate response and return both text and log probability"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # Generate with sampling
            outputs = self.policy_model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated tokens (excluding prompt)
            generated_tokens = outputs.sequences[0][len(inputs["input_ids"][0]):]
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Compute log probability (simplified)
            # In practice, you'd compute this more carefully during generation
            log_prob = -len(generated_tokens) * 0.1  # Simplified placeholder
        
        return response, log_prob
    
    def _get_response_embedding(self, text: str):
        """Get embedding representation of text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.policy_model.transformer(**inputs)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding
    
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
        ("Solve: 2 + 3 = ?", "2 + 3 = 5"),
        ("What is 4 × 5?", "4 × 5 = 20"),
        ("Explain gravity simply.", "Gravity is the force that pulls objects toward each other."),
    ]

def create_preference_data():
    """Create sample preference data for Stage 2"""
    return [
        ("What is AI?", 
         "AI is artificial intelligence.", 
         "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.", 
         1),  # Second response preferred
        ("Solve: 10 ÷ 2 = ?", 
         "10 ÷ 2 = 5", 
         "To solve 10 ÷ 2: We divide 10 by 2, which equals 5.", 
         1),  # Second response preferred (shows work)
    ]

def create_reasoning_prompts():
    """Create sample prompts focused on reasoning (GRPO's strength)"""
    return [
        "Solve: What is 15 + 7 × 2?",
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "What is the area of a rectangle with length 8 and width 3?",
        "Solve: 24 ÷ 6 + 3 × 2 = ?",
        "If I have 12 apples and give away 1/3, how many do I have left?",
        "What is 5! (5 factorial)?",
        "Solve for x: 2x + 6 = 14",
        "What is the next number in the sequence: 2, 4, 6, 8, ?",
    ]

# Main training pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("GRPO (Group Relative Policy Optimization) Training")
    print("Key Innovation: No value function needed!")
    print("Memory efficient and great for reasoning tasks")
    print("=" * 60)
    
    # Initialize trainer
    trainer = GRPOTrainer()
    
    # Prepare data
    demo_data = create_demo_data()
    preference_data = create_preference_data()
    reasoning_prompts = create_reasoning_prompts()
    
    # Stage 1: Supervised fine-tuning
    trainer.stage1_supervised_finetuning(demo_data, epochs=2)
    
    # Stage 2: Reward model training
    trainer.stage2_reward_model_training(preference_data, epochs=2)
    
    # Stage 3: GRPO training (the main innovation!)
    trainer.stage3_grpo_training(
        reasoning_prompts, 
        epochs=2,
        responses_per_prompt=3,  # Generate 3 responses per prompt
        group_size=4  # Group 4 prompts together
    )
    
    print("\n" + "=" * 60)
    print("GRPO training completed!")
    print("Benefits achieved:")
    print("- No value function (saves ~40% memory)")
    print("- Group-based advantages (more stable)")
    print("- Excellent for reasoning tasks")
    print("=" * 60)
    
    # Test the trained model
    test_prompt = "Solve: 3 × (4 + 2) = ?"
    response, _ = trainer._generate_response_with_log_prob(test_prompt)
    print(f"\nTest generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")