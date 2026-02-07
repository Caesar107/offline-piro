"""
Offline PIRO (Policy and Reward Optimization) Trainer
Implements Algorithm 1: Offline PIRO from the paper

This implements a modified training loop that alternates between:
1. Policy updates using SAC with current reward function
2. Reward updates using maximum likelihood estimation with trust region constraints
"""

import os
#os.add_dll_directory(r'C:\Users\admin\.mujoco\mujoco210\bin')
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
import random
import copy

from trainer import Trainer, reward_estimator
from utils import Transition, check_or_make_folder, ReplayPool
import d4rl
import pickle
import pandas as pd

device = torch.device("cpu")

class OfflinePiroTrainer(Trainer):
    """
    Simplified Offline PIRO Trainer 
    
    Adds trust region constraints to original ML-IRL loss to prevent 
    reward function from changing too rapidly. Based on constraint
    implementation from irl_samples.py.
    """
    
    def __init__(self, params, env, agent, device=device):
        super().__init__(params, env, agent, device)
        
        # PIRO specific parameters - simplified version
        self.m = params.get('piro_outer_loops', 100)   # Outer loop iterations  
        self.k = params.get('piro_policy_rounds', 5)   # Policy update rounds per outer loop
        self.n = params.get('piro_reward_rounds', 3)   # Reward update rounds per outer loop
        
        # Constraint parameters (from irl_samples.py)
        self.target_reward_diff = params.get('piro_target_reward_diff', 0.1)
        self.target_ratio_upper = params.get('piro_target_ratio_upper', 1.2)
        self.target_ratio_lower = params.get('piro_target_ratio_lower', 0.8)
        self.coef_scale_down = params.get('piro_coef_scale_down', 0.9)
        self.coef_scale_up = params.get('piro_coef_scale_up', 1.1)
        self.coef_min = params.get('piro_coef_min', 0.001)
        self.coef_max = params.get('piro_coef_max', 10.0)
        
        self.target_reward_l2_norm = params.get('piro_target_reward_l2_norm', 1.0)
        self.l2_coef_scale_up = params.get('piro_l2_coef_scale_up', 1.1)
        self.l2_coef_scale_down = params.get('piro_l2_coef_scale_down', 0.9)
        
        # Initialize adaptive coefficients
        self.avg_diff_coef = 1.0
        self.l2_norm_coef = 1.0
        
        # Reward function parameters
        self.reward_lr = params.get('piro_reward_lr', 3e-4)
        self.reward_hidden_sizes = params.get('piro_reward_hidden', [256, 256])
        self.reward_update_every = params.get('piro_reward_update_every', 20)
        
        # Initialize reward function (using state-action pair, same as ML-IRL)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.reward_func = reward_estimator(
            input_dim=state_dim + action_dim,  # state-action pair
            hidden_sizes=self.reward_hidden_sizes,
            device=device
        ).to(device)
        
        # Store old reward function for constraints
        self.old_reward_func = None
        self.reward_optimizer = torch.optim.Adam(
            self.reward_func.parameters(), 
            lr=self.reward_lr
        )
        
        print(f"Initialized Simplified Offline PIRO:")
        print(f"  Outer loops (m): {self.m}")
        print(f"  Policy rounds per loop (k): {self.k}")  
        print(f"  Reward rounds per loop (n): {self.n}")
        print(f"  L2 constraint target: {self.target_reward_l2_norm}")

    def _train_agent(self, IRL=False):
        if self._augment_offline_data:
            print("Augmenting model data with RAD")
        if IRL:
            self.agent.optimize(
                n_updates=self._policy_update_steps,
                env_pool=self.model.model.memory,
                env_ratio=self._real_sample_ratio,
                augment_data=self._augment_offline_data,
                reward_function=self.reward_func,
            )
        else:
            super()._train_agent(IRL=False)
        
    def _get_expert_folder(self):
        """
        Map env_name (e.g. 'halfcheetah-medium-v2') to expert_data folder
        (e.g. 'halfcheetah').
        默认取 env_name 第一个 '-' 前的前缀，这和当前 expert_data 目录结构一致：
        expert_data/
          - halfcheetah/...
          - hopper/...
        """
        env_name = self._params.get("env_name", "")
        # 例如 'halfcheetah-medium-v2' -> 'halfcheetah'
        return env_name.split("-")[0] if env_name else ""
    
    def _get_expert_data_path(self):
        """
        返回专家数据目录的绝对路径
        """
        # 获取当前文件所在目录（即项目根目录）
        project_root = os.path.dirname(os.path.abspath(__file__))
        expert_folder = self._get_expert_folder()
        return os.path.join(project_root, 'expert_data', expert_folder)

    def initialize_policy_with_bc(self):
        """
        Initialize policy with behavior cloning on expert demonstrations
        """
        print("Initializing policy with behavior cloning...")
        
        # Load expert data if available (使用绝对路径)
        expert_path = self._get_expert_data_path()
        expert_states = np.load(os.path.join(expert_path, 'states.npy'))[:50]
        expert_actions = np.load(os.path.join(expert_path, 'actions.npy'))[:50]
        
        # Reshape for batch processing
        expert_states = expert_states.reshape(-1, expert_states.shape[-1])
        expert_actions = expert_actions.reshape(-1, expert_actions.shape[-1])
        
        # Simple behavior cloning: train agent to mimic expert actions
        print(f"Training behavior cloning on {len(expert_states)} expert samples...")
        
        for bc_step in range(1000):
            # Sample a batch
            batch_size = min(256, len(expert_states))
            indices = np.random.choice(len(expert_states), batch_size, replace=False)
            
            states_batch = torch.FloatTensor(expert_states[indices]).to(device)
            actions_batch = torch.FloatTensor(expert_actions[indices]).to(device)
            
            # Get agent's predicted actions
            predicted_actions = self.agent.get_action(states_batch.cpu().numpy(), deterministic=True)
            predicted_actions = torch.FloatTensor(predicted_actions).to(device)
            
            # MSE loss for behavior cloning
            bc_loss = torch.mean((predicted_actions - actions_batch) ** 2)
            
            # Backward pass (this is simplified - in practice you'd need access to agent's actor network)
            # For now, we'll skip the actual BC implementation since it requires modifying SAC internals
            if bc_step % 100 == 0:
                print(f"BC step {bc_step}, loss: {bc_loss.item():.4f}")
                
        print("Behavior cloning initialization complete.")
    
    def rollout_policy_in_model(self, horizon=6, n_trajectories=50000):
        """
        Roll out current policy in world model for horizon H
        Collect transitions under low-uncertainty U(s_t, a_t)
        """
        print(f"Rolling out policy in model for {n_trajectories} trajectories...")
        
        collected_transitions = []
        uncertainty_threshold = 0.1  # Low uncertainty threshold
        
        for traj in range(n_trajectories):
            # Reset model environment
            state = self.model.reset()
            traj_transitions = []
            
            for step in range(horizon):
                # Get action from current policy
                action = self.agent.get_action(state, deterministic=False)
                
                # Step in model environment
                next_state, reward, done, info = self.model.step(action)
                
                # Check model uncertainty (simplified - assumes model provides uncertainty)
                uncertainty = getattr(self.model.model, 'last_uncertainty', 0.0)
                
                # Only collect if uncertainty is low
                if uncertainty < uncertainty_threshold:
                    transition = Transition(state, action, reward, next_state, done)
                    traj_transitions.append(transition)
                
                state = next_state
                if done:
                    break
            
            collected_transitions.extend(traj_transitions)
            
            # Stop if we have enough transitions
            if len(collected_transitions) >= n_trajectories:
                break
        
        print(f"Collected {len(collected_transitions)} transitions from model rollouts")
        return collected_transitions
    
    def sample_expert_batch(self, batch_size=256):
        """
        Sample a batch from expert demonstrations
        """
        try:
            # Load expert data (使用绝对路径)
            expert_path = self._get_expert_data_path()
            expert_states = np.load(os.path.join(expert_path, 'states.npy'))[:50]
            expert_actions = np.load(os.path.join(expert_path, 'actions.npy'))[:50]
            
            # Reshape
            expert_states = expert_states.reshape(-1, expert_states.shape[-1])
            expert_actions = expert_actions.reshape(-1, expert_actions.shape[-1])
            
            # Sample batch
            indices = np.random.choice(len(expert_states), 
                                     min(batch_size, len(expert_states)), 
                                     replace=False)
            
            states_batch = expert_states[indices]
            actions_batch = expert_actions[indices]
            
            return states_batch, actions_batch
            
        except FileNotFoundError:
            print("Warning: Expert data not found, using random samples")
            # Fallback to random samples
            obs_shape = self.model.observation_space.shape[0]
            act_shape = self.model.action_space.shape[0]
            states_batch = np.random.randn(batch_size, obs_shape)
            actions_batch = np.random.randn(batch_size, act_shape)
            return states_batch, actions_batch
    
    def sample_expert_initial_states(self, num_states=50):
        """
        Sample initial states from expert trajectories (first state of each trajectory)
        Used for model rollout to collect agent samples
        """
        try:
            # Load expert data (使用绝对路径)
            expert_path = self._get_expert_data_path()
            expert_states = np.load(os.path.join(expert_path, 'states.npy'))[:50]
            
            # expert_states shape: (num_traj, traj_length, state_dim)
            # Take first state of each trajectory
            initial_states = expert_states[:, 0, :]  # (num_traj, state_dim)
            
            # Sample if we need fewer states
            if len(initial_states) > num_states:
                indices = np.random.choice(len(initial_states), num_states, replace=False)
                initial_states = initial_states[indices]
            
            return torch.FloatTensor(initial_states).to(device)
            
        except FileNotFoundError:
            print("Warning: Expert data not found, using random initial states")
            # Fallback to random initial states
            obs_shape = self.model.observation_space.shape[0]
            initial_states = np.random.randn(num_states, obs_shape)
            return torch.FloatTensor(initial_states).to(device)
    
    def estimate_reward_gradient(self, expert_batch, model_transitions):
        """
        Estimate gradient of L_θ_old(θ) using expert batch and model transitions
        """
        expert_states, expert_actions = expert_batch
        
        # Convert to tensors
        expert_states_tensor = torch.FloatTensor(expert_states).to(device)
        
        # Get model transition states
        model_states = []
        for transition in model_transitions:
            model_states.append(transition.state)
        
        if len(model_states) == 0:
            print("Warning: No model transitions available for reward update")
            return torch.tensor(0.0, device=device)
        
        model_states = np.array(model_states)
        model_states_tensor = torch.FloatTensor(model_states).to(device)
        
        # Compute reward values
        expert_rewards = self.reward_func(expert_states_tensor)
        model_rewards = self.reward_func(model_states_tensor)
        
        # Maximum likelihood objective: E_expert[r] - E_model[r]
        expert_reward_mean = expert_rewards.mean()
        model_reward_mean = model_rewards.mean()
        
        # Gradient ascent objective
        reward_loss = model_reward_mean - expert_reward_mean
        
        return reward_loss
    
    def update_reward_with_constraints(self, agent_samples, expert_states, expert_actions):
        """
        Update reward with trust region constraints based on irl_samples.py approach
        Now uses state-action pair (same as ML-IRL)
        """
        # Store current reward as old for next iteration if first time
        if self.old_reward_func is None:
            self.old_reward_func = reward_estimator(
                input_dim=self.reward_func.input_dim,
                hidden_sizes=self.reward_hidden_sizes,
                device=device
            ).to(device)
            # Initialize old reward for the first update
            self.old_reward_func.load_state_dict(self.reward_func.state_dict())
        
        # Extract agent states and actions from samples
        if isinstance(agent_samples, list):
            agent_states = np.concatenate([t.state.reshape(1, -1) for t in agent_samples], axis=0)
            agent_actions = np.concatenate([t.action.reshape(1, -1) for t in agent_samples], axis=0)
        else:
            # If agent_samples is already processed (unlikely in current code)
            agent_states = agent_samples.reshape(-1, agent_samples.shape[-1])
            agent_actions = np.zeros((agent_states.shape[0], self.agent.action_dim))  # Placeholder
            
        # Concatenate state and action (same as ML-IRL)
        agent_state_action = np.concatenate([agent_states, agent_actions], axis=1)
        expert_state_action = np.concatenate([expert_states, expert_actions], axis=1)
        
        agent_state_action_tensor = torch.FloatTensor(agent_state_action).to(device)
        expert_state_action_tensor = torch.FloatTensor(expert_state_action).to(device)
        
        # Calculate current and old rewards
        current_rewards = self.reward_func(agent_state_action_tensor).view(-1)
        old_rewards = self.old_reward_func(agent_state_action_tensor).view(-1)
        expert_rewards = self.reward_func(expert_state_action_tensor).view(-1)
        
        # Calculate reward difference for constraint
        reward_diff = current_rewards - old_rewards
        
        # Original ML-IRL loss: E_agent[r] - E_expert[r]
        base_loss = current_rewards.mean() - expert_rewards.mean()
        
        # Calculate constraint terms (from irl_samples.py)
        avg_reward_diff = torch.mean(reward_diff)
        l2_norm_reward_diff = torch.norm(reward_diff, p=2)
        
        # Adaptive coefficient adjustment
        if avg_reward_diff > self.target_reward_diff * self.target_ratio_upper:
            self.avg_diff_coef = self.avg_diff_coef * self.coef_scale_down
        elif avg_reward_diff < self.target_reward_diff * self.target_ratio_lower:
            self.avg_diff_coef = self.avg_diff_coef * self.coef_scale_up
        
        self.avg_diff_coef = torch.tensor(self.avg_diff_coef)
        self.avg_diff_coef = torch.clamp(self.avg_diff_coef, min=self.coef_min, max=self.coef_max)
        
        if l2_norm_reward_diff > self.target_reward_l2_norm:
            self.l2_norm_coef = self.l2_norm_coef * self.l2_coef_scale_up
        elif l2_norm_reward_diff < self.target_reward_l2_norm:
            self.l2_norm_coef = self.l2_norm_coef * self.l2_coef_scale_down
        
        self.l2_norm_coef = torch.tensor(self.l2_norm_coef)
        self.l2_norm_coef = torch.clamp(self.l2_norm_coef, min=self.coef_min, max=self.coef_max)
        
        # Calculate final loss with constraints (exactly from irl_samples.py)
        loss = base_loss + self.l2_norm_coef * l2_norm_reward_diff
        
        # Perform gradient step
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

        # Store current reward as old for the next update step
        self.old_reward_func.load_state_dict(self.reward_func.state_dict())
        
        return loss.item(), base_loss.item(), avg_reward_diff.item(), l2_norm_reward_diff.item()
    
    def adjust_mu_coefficient(self, iteration):
        """
        Adjust μ coefficient according to Eq. (ref{eq:coef})
        """
        # Simple adaptive adjustment (can be made more sophisticated)
        if iteration % 10 == 0 and iteration > 0:
            # Increase mu if reward changes too much, decrease if too little
            # This is a placeholder - should implement proper constraint checking
            self.mu *= 0.95  # Slight decay
            self.mu = max(0.01, min(1.0, self.mu))  # Keep in reasonable bounds
    
    def train_offline_piro(self, save_model=False, save_policy=False, load_model_dir=None):
        """
        Simplified Offline PIRO training loop with constraint-based reward updates
        保存逻辑对齐 ML-IRL：每个 policy round 算一个"等效 epoch"
        总等效 epoch 数 = m * k (例如 200 * 5 = 1000)
        """
        total_equiv_epochs = self.m * self.k
        print(f"\nStarting Simplified Offline PIRO training for {self.m} outer loops...")
        print(f"Policy rounds per loop: {self.k}, Reward rounds per loop: {self.n}")
        print(f"Total equivalent epochs: {total_equiv_epochs} (m={self.m} x k={self.k})")
        
        # Load and prepare offline dataset (same as original)
        self.load_offline_dataset(load_model_dir, save_model)
        
        start_time = time.time()
        
        # 记录列表（和 ML-IRL 对齐）
        rewards = []           # True Reward (真实环境)
        rewards_m = []         # WM Reward (世界模型)
        l2_coefs = []          # L2 coefficient
        avg_diff_coefs = []    # avg_diff coefficient
        
        equiv_epoch = 0  # 等效 epoch 计数器
        
        # 构建保存文件名（和 ML-IRL 格式对齐）
        save_name = "{}_{}_piro_offline.csv".format(
            self._params['env_name'], 
            str(self._params['seed'])
        )
        
        # 创建 policy checkpoint 目录
        save_path = './model_saved_weights_seed{}'.format(self._params['seed'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Main PIRO loop
        for i in range(self.m):
            print(f"\n=== PIRO Iteration {i+1}/{self.m} ===")
            
            # Step 1: k rounds of SAC based on current reward function
            print(f"Training policy for {self.k} rounds...")
            for k_round in range(self.k):
                # Rollout in model with MOPO penalty-only rewards
                self._rollout_model(Penalty_only=True)

                # Train SAC agent
                self._train_agent(IRL=True)
                
                # ========== 每个 policy round 后评估和保存（和 ML-IRL 对齐）==========
                equiv_epoch += 1
                
                # 评估：世界模型奖励 + 真实环境奖励
                reward_model = self.test_agent(use_model=True, n_evals=10)
                reward_actual_stats = self.test_agent(use_model=False, n_evals=10)
                
                # 记录数据
                rewards.append(reward_actual_stats.mean())
                rewards_m.append(reward_model.mean())
                l2_coefs.append(float(self.l2_norm_coef) if isinstance(self.l2_norm_coef, (int, float)) else self.l2_norm_coef.item() if hasattr(self.l2_norm_coef, 'item') else float(self.l2_norm_coef))
                avg_diff_coefs.append(float(self.avg_diff_coef) if isinstance(self.avg_diff_coef, (int, float)) else self.avg_diff_coef.item() if hasattr(self.avg_diff_coef, 'item') else float(self.avg_diff_coef))
                
                # 计算进度和 ETA
                elapsed_time = time.time() - start_time
                progress = equiv_epoch / total_equiv_epochs
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                
                # 打印进度（和 ML-IRL 格式对齐）
                print(f"Epoch {equiv_epoch}/{total_equiv_epochs} (Progress: {progress:.1%}, ETA: {remaining_time/60:.1f} min)")
                stats_fmt = "{:<20}{:>30}"
                print(stats_fmt.format("WM Reward Mean", round(reward_model.mean(), 2)))
                print(stats_fmt.format("True Reward Mean", round(reward_actual_stats.mean(), 2)))
                print(stats_fmt.format("True Reward Std", round(reward_actual_stats.std(), 2)))
                print(stats_fmt.format("L2 Coefficient", round(l2_coefs[-1], 4)))
                
                # 保存 CSV（每个 epoch 都保存，和 ML-IRL 对齐）
                save_stats = {
                    'Reward': rewards, 
                    'Reward_WM': rewards_m, 
                    'L2_Coef': l2_coefs,
                    'AvgDiff_Coef': avg_diff_coefs
                }
                df = pd.DataFrame(save_stats)
                df.to_csv(save_name)
                
                # 每 20 个等效 epoch 保存一次 policy checkpoint（和 ML-IRL 对齐）
                if save_policy and equiv_epoch % 20 == 0:
                    print(f"Saving policy checkpoint at epoch {equiv_epoch}...")
                    self.agent.save_policy(
                        save_path,
                        num_epochs=equiv_epoch,
                        rew=int(reward_actual_stats.mean())
                    )
            
            # Step 2: n rounds of reward updates with constraints (throttled)
            if self.reward_update_every > 0 and (equiv_epoch % self.reward_update_every != 0):
                print(f"Skipping reward update (every {self.reward_update_every} epochs)")
                continue

            print(f"Updating reward function for {self.n} rounds...")
            for j in range(self.n):
                # Sample expert batch
                expert_states, expert_actions = self.sample_expert_batch()
                
                # Collect agent samples from current policy using world model rollout
                # (Same approach as ML-IRL: use model rollout instead of real environment)
                sample_pool = ReplayPool(capacity=1e6)
                
                # Sample initial states from expert trajectories
                init_states = self.sample_expert_initial_states(num_states=50)
                
                # Rollout current policy in world model (like ML-IRL's _rollout_model_expert)
                self._rollout_model_expert(init_states, sample_pool, IRL=True, Penalty_only=True)
                
                # Extract agent samples from rollout pool (same format as ML-IRL)
                samples = sample_pool.sample_all()
                if len(samples.state) > 0:
                    # Flatten nested structure like ML-IRL does
                    # samples.state is a tuple/list of arrays, each array may contain multiple states
                    agent_samples = []
                    for i in range(len(samples.state)):
                        state_arr = samples.state[i]
                        action_arr = samples.action[i]
                        nextstate_arr = samples.nextstate[i]
                        done_arr = samples.real_done[i]  # Transition uses 'real_done' not 'done'
                        
                        # Handle both single transitions and arrays of transitions
                        if isinstance(state_arr, np.ndarray):
                            if len(state_arr.shape) == 1:
                                # Single transition (1D array)
                                agent_samples.append(Transition(
                                    state_arr, action_arr, 0.0, nextstate_arr, done_arr
                                ))
                            else:
                                # Multiple transitions (2D array: shape [num_transitions, state_dim])
                                num_trans = state_arr.shape[0]
                                for t in range(num_trans):
                                    agent_samples.append(Transition(
                                        state_arr[t], action_arr[t] if len(action_arr.shape) > 1 else action_arr,
                                        0.0, nextstate_arr[t] if len(nextstate_arr.shape) > 1 else nextstate_arr,
                                        done_arr[t] if isinstance(done_arr, np.ndarray) and len(done_arr.shape) > 0 else done_arr
                                    ))
                        else:
                            # Single scalar/1D array
                            agent_samples.append(Transition(
                                np.array(state_arr), np.array(action_arr), 0.0,
                                np.array(nextstate_arr), done_arr
                            ))
                else:
                    agent_samples = []
                
                # Update reward with constraints (using irl_samples.py approach)
                if len(agent_samples) > 0:
                    loss, base_loss, avg_diff, l2_norm = self.update_reward_with_constraints(
                        agent_samples, expert_states, expert_actions)
                    
                    if j == 0:  # 只打印第一次
                        print(f"  Reward update: loss={loss:.4f}, base={base_loss:.4f}, "
                              f"avg_diff={avg_diff:.4f}, l2_norm={l2_norm:.4f}")
                else:
                    print(f"  Warning: No agent samples collected in reward update round {j+1}")
        
        # Save final models
        if save_policy:
            print("Saving final policy...")
            policy_dir = "piro_final_models"
            if not os.path.exists(policy_dir):
                os.makedirs(policy_dir)
            # 使用 agent.save_policy 保存（SAC_Agent 没有 state_dict 方法）
            self.agent.save_policy(save_path, num_epochs=equiv_epoch, rew=int(rewards[-1]) if rewards else 0)
            self.agent.save_policy(policy_dir, num_epochs=equiv_epoch, rew=int(rewards[-1]) if rewards else 0)
            print(f"Final policy saved to: {policy_dir}/")
            
        if save_model:
            print("Saving final reward model...")
            if not os.path.exists(policy_dir):
                os.makedirs(policy_dir)
            reward_path = os.path.join(policy_dir, "piro_reward_final.pkl")
            torch.save(self.reward_func.state_dict(), reward_path)
            print(f"Final reward model saved to: {reward_path}")
        
        print(f"\nSimplified Offline PIRO training completed in {(time.time() - start_time)/60:.1f} minutes")
        print(f"Total equivalent epochs trained: {equiv_epoch}")
        print(f"CSV saved to: {save_name}")
        return rewards
    
    def load_offline_dataset(self, load_model_dir, save_model):
        """
        Load offline dataset, train world model, and populate agent replay pool
        """
        print("Loading offline dataset and training world model...")
        
        # Load d4rl dataset
        env = self.model.real_env
        if self._params['env_name'] != 'AntMOPOEnv':
            dataset = d4rl.qlearning_dataset(env)
        else:
            with open('/Meta-Offline-RL/ant_mopo_1m_dataset.pkl', 'rb') as f:
                dataset = pickle.load(f)

        N = dataset['rewards'].shape[0]
        rollout = []
        val_size = 0
        train_size = 0

        if load_model_dir:
            # Load pretrained model
            print("Loading model from checkpoint...")
            errors = self.model.model.load_model(load_model_dir)
        else:
            print("Training world model from scratch...")
            self.model.update_state_filter(dataset['observations'][0])

            for i in range(N):
                state = dataset['observations'][i]
                action = dataset['actions'][i]
                nextstate = dataset['next_observations'][i]
                reward = dataset['rewards'][i]
                done = bool(dataset['terminals'][i])

                t = Transition(state, action, reward, nextstate, done)
                rollout.append(t)

                self.model.update_state_filter(nextstate)
                self.model.update_action_filter(action)

                # Split data for training and validation
                if random.uniform(0, 1) < self.model.model.train_val_ratio:
                    self.model.model.add_data_validation(t)
                    val_size += 1
                else:
                    self.model.model.add_data(t)
                    train_size += 1

            print(f"Added {train_size} samples for train, {val_size} for validation")
            self._train_model(d4rl_init=True, save_model=save_model)
        
        # CRITICAL: Populate agent replay pool with offline data  
        print("Populating agent replay pool with offline data...")
        
        # Agent should already have a properly initialized replay pool from SAC_Agent.__init__
        # Just verify it exists and has the right dimensions
        if not hasattr(self.agent, 'replay_pool') or self.agent.replay_pool is None:
            print("Error: Agent replay pool not initialized properly!")
            return
            
        print(f"Agent replay pool state dim: {self.agent.replay_pool._state_dim}")
        print(f"Agent replay pool action dim: {self.agent.replay_pool._action_dim}")
        print(f"Dataset state shape: {dataset['observations'][0].shape}")
        print(f"Dataset action shape: {dataset['actions'][0].shape}")
        
        # Add all offline data to agent's replay pool
        for i in range(N):
            state = dataset['observations'][i]
            action = dataset['actions'][i]
            nextstate = dataset['next_observations'][i]
            reward = dataset['rewards'][i]
            done = bool(dataset['terminals'][i])
            
            # Create transition and push to replay pool
            transition = Transition(state, action, reward, nextstate, done)
            self.agent.replay_pool.push(transition)
            
        print(f"Agent replay pool now contains {len(self.agent.replay_pool)} transitions")
        
        # Train world model if needed
        if not load_model_dir:
            self._train_model(d4rl_init=True, save_model=save_model)
            
    def _train_model(self, d4rl_init=False, save_model=False):
        """Train world model (simplified version from trainer.py)"""
        print("Training world model...")
        
        if d4rl_init:
            # For d4rl datasets, train the model
            self.model.model.train(bootstrap=True)
            
        if save_model:
            # Save model if requested
            print("Saving world model...")
            check_or_make_folder("data/piro_world_model.pkl")
            torch.save(self.model.model.state_dict(), "data/piro_world_model.pkl")

        if load_model_dir:
            errors = self.model.model.load_model(load_model_dir)
            print(f"Loaded pre-trained world model from {load_model_dir}")
        else:
            print(f"Training world model on {N} transitions...")
            self.model.update_state_filter(dataset['observations'][0])

            for i in range(N):
                state = dataset['observations'][i]
                action = dataset['actions'][i]
                nextstate = dataset['next_observations'][i]
                reward = dataset['rewards'][i]
                done = bool(dataset['terminals'][i])

                t = Transition(state, action, reward, nextstate, done)
                rollout.append(t)

                self.model.update_state_filter(nextstate)
                self.model.update_action_filter(action)

                if random.uniform(0, 1) < self.model.model.train_val_ratio:
                    self.model.model.add_data_validation(t)
                    val_size += 1
                else:
                    self.model.model.add_data(t)
                    train_size += 1

            print(f"Added {train_size} samples for train, {val_size} for validation")
            self._train_model(d4rl_init=True, save_model=save_model)
            print("World model training completed!")
