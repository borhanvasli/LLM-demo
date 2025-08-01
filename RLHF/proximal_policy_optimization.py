import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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
    RLHF implementation with Value Function for variance reduction.
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
        
        # NEW: Value function for advantage estimation
        self.value_function = ValueFunction(self.policy_model.config.hidden_size)
        
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
    
    def stage3_ppo_with_value_function(self, prompts: List[str], epochs=3, lr=1e-5, 
                                      kl_coeff=0.1, value_lr=5e-4, gamma=0.99):
        """
        Stage 3: PPO training with Value Function for advantage estimation
        This is much more stable than the previous simple policy gradient approach
        """
        print("\nStage 3: PPO Training with Value Function...")
        
        policy_optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        value_optimizer = optim.AdamW(self.value_function.parameters(), lr=value_lr)
        
        for epoch in range(epochs):
            total_reward = 0
            total_advantage = 0
            total_value_loss = 0
            
            for prompt in prompts:
                # === EXPERIENCE COLLECTION ===
                # Generate response and collect trajectory data
                trajectory = self._collect_trajectory(prompt)
                
                states = trajectory['states']
                actions = trajectory['actions'] 
                rewards = trajectory['rewards']
                log_probs = trajectory['log_probs']
                
                # === VALUE FUNCTION PREDICTIONS ===
                # Get value estimates for all states in trajectory
                state_embeddings = torch.stack([self._get_response_embedding(state) for state in states])
                values = self.value_function(state_embeddings).squeeze(-1)
                
                # === ADVANTAGE COMPUTATION ===
                # This is the KEY improvement over simple policy gradient
                advantages, returns = self._compute_advantages_and_returns(
                    rewards, values, gamma=gamma
                )
                
                # === POLICY UPDATE (Actor) ===
                # Normalize advantages for stability
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy gradient with advantage weighting
                policy_loss = -(torch.tensor(log_probs) * normalized_advantages).mean()
                
                # Add KL penalty
                kl_penalty = self._calculate_kl_penalty_batch(states)
                total_policy_loss = policy_loss + kl_coeff * kl_penalty
                
                # Update policy
                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                policy_optimizer.step()
                
                # === VALUE FUNCTION UPDATE (Critic) ===
                # Train value function to predict returns accurately
                value_loss = F.mse_loss(values, returns)
                
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                
                # === LOGGING ===
                total_reward += sum(rewards)
                total_advantage += advantages.mean().item()
                total_value_loss += value_loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Avg Reward: {total_reward/len(prompts):.4f}")
            print(f"  Avg Advantage: {total_advantage/len(prompts):.4f}")
            print(f"  Value Loss: {total_value_loss/len(prompts):.4f}")
    
    def _collect_trajectory(self, prompt: str, max_tokens=50):
        """
        Generate a complete trajectory (sequence of states, actions, rewards)
        This is essential for proper advantage computation
        """
        states = [prompt]  # Start with prompt
        actions = []
        rewards = []
        log_probs = []
        
        current_text = prompt
        
        for step in range(max_tokens):
            # Get next token distribution
            inputs = self.tokenizer(current_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.policy_model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Sample action (next token)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[action]).item()
                
                # Decode action to text
                action_text = self.tokenizer.decode([action])
                current_text += action_text
                
                # Store trajectory data
                states.append(current_text)
                actions.append(action)
                log_probs.append(log_prob)
                
                # Compute reward for this step (sparse reward at end, 0 elsewhere)
                if step == max_tokens - 1 or action == self.tokenizer.eos_token_id:
                    # Final reward from reward model
                    embedding = self._get_response_embedding(current_text)
                    final_reward = self.reward_model(embedding.unsqueeze(0)).item()
                    rewards.append(final_reward)
                    break
                else:
                    rewards.append(0.0)  # No intermediate rewards
        
        return {
            'states': states[:-1],  # Remove last state (no action taken)
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs
        }
    
    def _compute_advantages_and_returns(self, rewards, values, gamma=0.99):
        """
        Compute advantages and returns using the Value Function
        
        This is the CORE benefit of having a value function:
        - Advantage = How much better/worse was this action than expected?
        - Return = Discounted future rewards (what we're trying to predict)
        
        Args:
            rewards: List of rewards at each step
            values: Tensor of value function predictions
            gamma: Discount factor
            
        Returns:
            advantages: How much better/worse each action was than expected
            returns: Discounted cumulative rewards (targets for value function)
        """
        rewards = torch.tensor(rewards, dtype=torch.float32)
        T = len(rewards)
        
        # === COMPUTE RETURNS (Discounted Future Rewards) ===
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        # Work backwards from the end
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # === COMPUTE ADVANTAGES ===
        # Advantage = Actual Return - Expected Return (Value Function)
        # This tells us: "Was this action better or worse than we expected?"
        
        if len(values) > len(returns):
            values = values[:len(returns)]  # Trim if necessary
        elif len(values) < len(returns):
            # Pad with zeros if value function didn't predict all states
            padding = torch.zeros(len(returns) - len(values))
            values = torch.cat([values, padding])
        
        advantages = returns - values.detach()  # Detach to avoid training policy on value gradients
        
        return advantages, returns
    
    def _calculate_kl_penalty_batch(self, states):
        """Calculate KL penalty for a batch of states"""
        total_kl = 0
        for state in states:
            kl = self._calculate_kl_penalty(state)
            total_kl += kl
        return total_kl / len(states)
    
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

class ValueFunction(nn.Module):
    """
    Value Function Network - Predicts expected future rewards
    
    GOAL: Learn V(s) = E[sum of future rewards | starting from state s]
    
    Why do we need this?
    1. Reduces variance in policy gradients (makes training more stable)
    2. Provides baseline for advantage computation  
    3. Helps the policy focus on truly good/bad actions
    """
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Single scalar output (expected return)
        )
    
    def forward(self, state_embedding):
        """
        Args:
            state_embedding: Vector representation of current state
            
        Returns:
            value: Scalar prediction of expected future reward
        """
        return self.layers(state_embedding)

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
    
    # Stage 3: PPO training with value function
    trainer.stage3_ppo_with_value_function(prompts, epochs=2)
    
    print("\nRLHF training completed!")
    
    # Test the trained model
    test_prompt = "What is machine learning?"
    response = trainer._generate_response(test_prompt)
    print(f"\nTest generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")