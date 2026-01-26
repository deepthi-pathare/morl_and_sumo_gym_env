"""Multi-Objective PPO Algorithm."""

import time
from copy import deepcopy
from typing import List, Optional, Union
from typing_extensions import override

import random
import gymnasium as gym
import numpy as np
import torch as th
import wandb
from torch import nn, optim
from torch.distributions import Categorical

from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import layer_init, mlp

class PPOReplayBuffer:
    """Replay buffer for both continuous and discrete action spaces."""

    def __init__(
        self,
        size: int,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        reward_dim: int,
        device: Union[th.device, str],
        discrete_actions: bool = True,
    ):
        """Initialize the replay buffer.

        Args:
            size: Buffer size
            num_envs: Number of environments (for VecEnv)
            obs_shape: Observation shape
            action_shape: Action shape - for discrete: number of actions or (n_actions,)
            reward_dim: Reward dimension
            device: Device where the tensors are stored
            discrete_actions: Whether the action space is discrete
        """
        self.size = size
        self.ptr = 0
        self.num_envs = num_envs
        self.device = device
        self.discrete_actions = discrete_actions
        
        self.obs = th.zeros((self.size, self.num_envs) + obs_shape).to(device)
        self.logprobs = th.zeros((self.size, self.num_envs)).to(device)
        self.rewards = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)
        self.dones = th.zeros((self.size, self.num_envs)).to(device)
        self.values = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)
        
        # Handle action tensor initialization based on action space type
        if discrete_actions:
            # For discrete actions, store as integer indices (no additional dimensions)
            self.actions = th.zeros((self.size, self.num_envs), dtype=th.long).to(device)
            self.action_masks = th.zeros((self.size, self.num_envs, action_shape), dtype=th.bool).to(device)
        else:
            # For continuous actions, preserve the original shape
            self.actions = th.zeros((self.size, self.num_envs) + action_shape).to(device)
            self.action_masks = th.ones((self.size, self.num_envs, action_shape), dtype=th.bool).to(device)

        self.weights = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)

    def add(self, obs, actions, logprobs, rewards, dones, values, weights, action_masks):
        """Add a bunch of new transition to the buffer. (VecEnv makes more transitions at once).

        Args:
            obs: Observations
            actions: Actions (integers for discrete, vectors for continuous)
            logprobs: Log probabilities of the actions
            rewards: Rewards
            dones: Done signals
            values: Values
        """
        self.obs[self.ptr] = obs
        
        if self.discrete_actions:
            # For discrete actions, ensure actions are stored as integers
            self.actions[self.ptr] = actions.long() if not actions.dtype == th.long else actions
        else:
            self.actions[self.ptr] = actions
            
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.weights[self.ptr] = weights
        self.action_masks[self.ptr] = action_masks
        self.ptr = (self.ptr + 1) % self.size

    def get(self, step: int):
        """Get data from the buffer at a specific step.

        Args:
            step: step

        Returns: A tuple of (obs, actions, logprobs, rewards, dones, values)
        """
        return (
            self.obs[step],
            self.actions[step],
            self.logprobs[step],
            self.rewards[step],
            self.dones[step],
            self.values[step],
            self.weights[step],
            self.action_masks[step],
        )

    def get_all(self):
        """Get all data from the buffer.

        Returns: A tuple of (obs, actions, logprobs, rewards, dones, values) containing all the data in the buffer.
        """
        return (
            self.obs,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values,
            self.weights,
            self.action_masks,
        )

def _hidden_layer_init(layer):
    layer_init(layer, weight_gain=np.sqrt(2), bias_const=0.0)

def _critic_init(layer):
    layer_init(layer, weight_gain=1.0)

def _value_init(layer):
    layer_init(layer, weight_gain=0.01)

class MOPPONet(nn.Module):
    """Actor-Critic network for discrete action spaces."""

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,  # For discrete: (n_actions,) or just n_actions
        reward_dim: int,
        net_arch: List = [64, 64],
        drop_rate: float = 0.01,
        layer_norm: bool = True,
    ):
        """Initialize the network.

        Args:
            obs_shape: Observation shape
            action_shape: Action shape - for discrete actions, this should be 
                         the number of discrete actions (int) or tuple with single element
            reward_dim: Reward dimension
            net_arch: Number of units per layer
            drop_rate: Dropout rate
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch
        
        # Handle action_shape for discrete case
        if isinstance(action_shape, (tuple, list)):
            self.n_actions = action_shape[0] if len(action_shape) == 1 else np.prod(action_shape)
        else:
            self.n_actions = action_shape

        # Weight conditioning features
        self.weights_features = mlp(reward_dim, -1, net_arch[:1])
        
        # State features
        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=net_arch[0])

        # Actor network: conditioned features -> action logits per reward dimension
        self.actor = mlp(
            input_dim=net_arch[0],  # Combined features dimension
            output_dim=self.n_actions * self.reward_dim,  # Actions X Rewards
            net_arch=net_arch[1:],
            drop_rate=drop_rate,
            layer_norm=layer_norm,
            activation_fn=nn.Tanh,
        )
        
        # Critic network: conditioned features -> multi-objective values (no weight conditioning)
        self.critic = mlp(
            input_dim=net_arch[0],
            output_dim=self.reward_dim,
            net_arch=net_arch[1:],
            drop_rate=drop_rate,
            layer_norm=layer_norm,
            activation_fn=nn.Tanh,
        )

        # Apply initialization
        self.actor.apply(_hidden_layer_init)
        self.critic.apply(_hidden_layer_init)
        _value_init(list(self.actor.modules())[-1])
        _critic_init(list(self.critic.modules())[-1])

    def _get_conditioned_features(self, obs, w):
        """Get conditioned features from observation and weights.
        
        Args:
            obs: Observations
            w: Weight vector for conditioning
            
        Returns:
            Combined features similar to QNet approach
        """
        sf = self.state_features(obs)  
        wf = self.weights_features(w)
        return sf * wf  # Element-wise multiplication

    def get_value(self, obs, w):
        """Get the value of an observation (not conditioned on weights).

        Args:
            obs: Observation
            w: Weight vector (ignored, kept for interface compatibility)

        Returns: The predicted multi-objective values of the observation.
        """
        conditioned_features = self._get_conditioned_features(obs, w)
        return self.critic(conditioned_features)

    def get_action_logits(self, obs, w, action_mask = None):
        """Get action logits conditioned on weights.
        
        Args:
            obs: Observation
            w: Weight vector for conditioning
            
        Returns:
            Action logits shaped as (batch_size, n_actions, reward_dim)
        """
        conditioned_features = self._get_conditioned_features(obs, w)
        logits = self.actor(conditioned_features)
        logits = logits.view(-1, self.n_actions, self.reward_dim) 
        
        if action_mask is not None:
            # Expand mask to match reward dimension: (batch, actions) -> (batch, actions, reward_dim)
            mask_expanded = action_mask.unsqueeze(-1).expand_as(logits)
            logits = th.where(
                mask_expanded.to(logits.device),
                logits,
                th.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            )
        return logits

    def get_action_and_value(self, obs, w, action_mask = None, action = None):
        """Get the action and value of an observation conditioned on weights.

        Args:
            obs: Observation
            w: Weight vector for conditioning
            action: Action. If None, a new action is sampled.

        Returns: A tuple of (action, logprob, entropy, value)
        """
        conditioned_features = self._get_conditioned_features(obs, w)
        
        # Get action logits with reward dimension
        logits = self.actor(conditioned_features)
        logits = logits.view(-1, self.n_actions, self.reward_dim)

        if action_mask is not None:
            # Expand mask to match reward dimension: (batch, actions) -> (batch, actions, reward_dim)
            mask_expanded = action_mask.unsqueeze(-1).expand_as(logits)
            logits = th.where(
                mask_expanded.to(logits.device),
                logits,
                th.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            )

        # Scalarize logits using weights to get final action logits
        # w can be either a single weight vector or a batch of weight vectors
        if w.dim() == 1:
            weighted_logits = th.einsum("r,bar->ba", w, logits)
        else:            
            weighted_logits = th.einsum("br,bar->ba", w, logits)
        distr = Categorical(logits=weighted_logits)

        if action is None:
            action = distr.sample()
        
        # Get multi-objective value 
        value = self.critic(conditioned_features)
        
        return (
            action,
            distr.log_prob(action),
            distr.entropy(),
            value,
        )

class MOPPO(MOPolicy):
    """ Multi-objective version of PPO.
    """

    def __init__(
        self,
        id: int,
        networks: MOPPONet,
        weights: np.ndarray,
        envs: gym.vector.SyncVectorEnv,
        log: bool = False,
        steps_per_iteration: int = 2048,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.995,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,#0.0,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.98,#0.95,
        device: Union[th.device, str] = "auto",
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ):
        """Multi-objective PPO for discrete action spaces.

        Args:
            id: Policy ID
            networks: Actor-Critic networks
            weights: Weights of the objectives
            envs: Vectorized environments
            log: Whether to log
            steps_per_iteration: Number of steps per iteration
            num_minibatches: Number of minibatches
            update_epochs: Number of epochs to update the network
            learning_rate: Learning rate
            gamma: Discount factor
            anneal_lr: Whether to anneal the learning rate
            clip_coef: PPO clipping coefficient
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            clip_vloss: Whether to clip the value loss
            max_grad_norm: Maximum gradient norm
            norm_adv: Whether to normalize the advantage
            target_kl: Target KL divergence
            gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda
            device: Device to use
            seed: Random seed
            rng: Random number generator
        """
        super().__init__(id, device)
        self.id = id
        self.envs = envs
        self.num_envs = envs.num_envs
        self.networks = networks
        self.device = device
        self.seed = seed
        if rng is not None:
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # PPO Parameters
        self.steps_per_iteration = steps_per_iteration
        self.np_weights = weights
        self.weights = th.from_numpy(weights.astype(np.float32)).to(self.device)
        self.batch_size = int(self.num_envs * self.steps_per_iteration)
        self.num_minibatches = num_minibatches
        self.minibatch_size = 64 #int(self.batch_size // num_minibatches)
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.anneal_lr = anneal_lr
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda
        self.log = log
        self.gae = gae

        self.optimizer = optim.Adam(networks.parameters(), lr=self.learning_rate, eps=1e-5)

        # Storage setup (the batch)
        self.batch = PPOReplayBuffer(
            self.steps_per_iteration,
            self.num_envs,
            self.networks.obs_shape,
            self.networks.action_shape,
            self.networks.reward_dim,
            self.device,
        )

    def __deepcopy__(self, memo):
        """Deepcopy method.

        Useful for genetic algorithms stuffs.
        """
        copied_net = deepcopy(self.networks)
        copied = type(self)(
            self.id,
            copied_net,
            self.weights.detach().cpu().numpy(),
            self.envs,
            self.log,
            self.steps_per_iteration,
            self.num_minibatches,
            self.update_epochs,
            self.learning_rate,
            self.gamma,
            self.anneal_lr,
            self.clip_coef,
            self.ent_coef,
            self.vf_coef,
            self.clip_vloss,
            self.max_grad_norm,
            self.norm_adv,
            self.target_kl,
            self.gae,
            self.gae_lambda,
            self.device,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate, eps=1e-5)
        copied.batch = deepcopy(self.batch)
        return copied

    def change_weights(self, new_weights: np.ndarray):
        """Change the weights of the scalarization function.

        Args:
            new_weights: New weights to apply.
        """
        self.np_weights = new_weights
        self.weights = th.from_numpy(deepcopy(new_weights).astype(np.float32)).to(self.device)

    def __extend_to_reward_dim(self, tensor: th.Tensor):
        # This allows to broadcast the tensor to match the additional dimension of rewards
        return tensor.unsqueeze(1).repeat(1, self.networks.reward_dim)

    def __collect_samples(self, obs: th.Tensor, done: th.Tensor, weight_support: List[np.ndarray]):
        """Fills the batch with {self.steps_per_iteration} samples collected from the environments.

        Args:
            obs: current observations
            done: current dones

        Returns:
            next observation and dones
        """
        np_w = np.repeat(self.np_weights[None, :], self.num_envs, axis=0)  # shape: (num_envs, reward_dim)
        w = self.weights.unsqueeze(0).repeat(self.num_envs, 1)             # torch.Tensor, same shape

        for step in range(0, self.steps_per_iteration):
            self.global_step += 1 * self.num_envs
            # Compute best action
            with th.no_grad():
                action_mask = self.get_action_mask(obs)
                action, logprob, _, value = self.networks.get_action_and_value(obs, w, action_mask)
                value = value.view(self.num_envs, self.networks.reward_dim)

            # For discrete actions, convert to numpy and ensure proper shape
            # action is already an integer tensor for discrete case
            action_np = action.cpu().numpy()
            
            # Perform action on the environment
            next_obs, reward, next_terminated, next_truncated, info = self.envs.step(action_np)
            reward = th.tensor(reward).to(self.device).view(self.num_envs, self.networks.reward_dim)

            # storing to batch
            self.batch.add(obs, action, logprob, reward, done, value, w, action_mask)

            # Next iteration
            obs = next_obs = th.Tensor(next_obs).to(self.device)
            done = th.Tensor(next_terminated | next_truncated).to(self.device) 

            # Terminates episodes
            env_term_indices = np.where(next_terminated | next_truncated)[0]
            for idx in env_term_indices:
                
                # Episode info logging
                if self.log and "episode" in info.keys():
                    # Reconstructs the dict by extracting the relevant information for each vectorized env
                    info_log = {k: v[idx] for k, v in info["episode"].items()}

                    log_episode_info(
                        info_log,
                        scalarization=np.dot,
                        #weights=self.weights, #TODO
                        weights=np_w[idx], #TODO
                        global_timestep=self.global_step,
                        id=self.id,
                        verbose = False
                    )

                # Change w for every episode. However store it before changing, 
                # in order to return latest used weight vectors when function is retuned.
                old_w = w 

                # Half time use current weight vector, half time sample new
                if random.random() < 0.5:
                    new_np_w = self.np_weights  # reuse current weight vector
                else:
                    new_np_w = random.choice(weight_support)  # sample new weight vector

                np_w[idx] = new_np_w
                w[idx] = th.tensor(new_np_w, dtype=th.float32, device=self.device)

                reset_obs, _ = self.envs.envs[idx].reset() 
                obs[idx] = th.tensor(reset_obs, device=self.device)

        return next_obs, done, old_w

    def get_value(self, obs, w):
        with th.no_grad():    
            value = self.networks.get_value(obs, w)
        return value

    def get_action_logits(self, obs, w):
        action_mask = self.get_action_mask(obs)
        with th.no_grad():    
            logits = self.networks.get_action_logits(obs, w, action_mask)
        return logits

    def __compute_advantages(self, next_obs, next_done, w):
        """Computes the advantages
        """
        with th.no_grad():
            next_value = self.networks.get_value(next_obs, w).reshape(self.num_envs, -1)
            if self.gae:
                advantages = th.zeros_like(self.batch.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_iteration)):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        _, _, _, _, done_t1, value_t1, _, _ = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        nextvalues = value_t1

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, value_t, _, _ = self.batch.get(t)
                    delta = reward_t + self.gamma * nextvalues * nextnonterminal - value_t
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.batch.values
            else:
                returns = th.zeros_like(self.batch.rewards).to(self.device)
                for t in reversed(range(self.steps_per_iteration)):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        _, _, _, _, done_t1, _, _, _ = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        next_return = returns[t + 1]

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, _, _, _ = self.batch.get(t)
                    returns[t] = reward_t + self.gamma * nextnonterminal * next_return
                advantages = returns - self.batch.values

        # Scalarization of the advantages
        advantages = (advantages * self.batch.weights).sum(dim=2)
        return returns, advantages

    @override
    def eval(self, obs: np.ndarray, w):
        """Returns the best action to perform based on the given obs and preference weights

        Returns:
            action as a numpy array (discrete action index)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0).repeat(self.num_envs, 1)  # duplicate observation to fit the NN input
        with th.no_grad():
            action_mask = self.get_action_mask(obs)
            action, _, _, _ = self.networks.get_action_and_value(obs, w, action_mask)

        # For discrete actions, return the action index as integer
        return action[0].detach().cpu().numpy().item()

    @override
    def update(self):
        # flatten the batch (b == batch)
        obs, actions, logprobs, _, _, values, weights, action_masks = self.batch.get_all()
        b_obs = obs.reshape((-1,) + self.networks.obs_shape)
        b_logprobs = logprobs.reshape(-1)
        
        # For discrete actions, actions are already integer indices
        # No need to reshape with action_shape for discrete case
        b_actions = actions.reshape(-1)
        
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1, self.networks.reward_dim)
        b_values = values.reshape(-1, self.networks.reward_dim)
        b_weights = weights.reshape(-1, self.networks.reward_dim)
        b_action_masks = action_masks.reshape(-1, self.networks.n_actions)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        # Perform multiple passes on the batch (that is shuffled every time)
        for epoch in range(self.update_epochs):
            self.np_random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                # mb == minibatch
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.networks.get_action_and_value(b_obs[mb_inds], b_weights[mb_inds], b_action_masks[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1, self.networks.reward_dim)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.networks.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record rewards for plotting purposes
        if self.log:
            wandb.log(
                {
                    f"charts_{self.id}/learning_rate": self.optimizer.param_groups[0]["lr"],
                    f"losses_{self.id}/value_loss": v_loss.item(),
                    f"losses_{self.id}/policy_loss": pg_loss.item(),
                    f"losses_{self.id}/entropy": entropy_loss.item(),
                    f"losses_{self.id}/old_approx_kl": old_approx_kl.item(),
                    f"losses_{self.id}/approx_kl": approx_kl.item(),
                    f"losses_{self.id}/clipfrac": np.mean(clipfracs),
                    f"losses_{self.id}/explained_variance": explained_var,
                    "global_step": self.global_step,
                },
            )

    def train(self, start_time, current_iteration: int, max_iterations: int, weight: np.ndarray, weight_support: List[np.ndarray]):
        """A training iteration: trains MOPPO for self.steps_per_iteration * self.num_envs.

        Args:
            start_time: time.time() when the training started
            current_iteration: current iteration number
            max_iterations: maximum number of iterations
            weight: Currently selected weight
            weight_support: Top K selected weights
        """
        self.change_weights(weight)

        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = th.Tensor(next_obs).to(self.device)  # num_envs x obs
        next_done = th.zeros(self.num_envs).to(self.device)
        # Annealing the rate if instructed to do so.
        if self.anneal_lr:
            frac = 1.0 - (current_iteration - 1.0) / max_iterations
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Fills buffer
        next_obs, next_done, w = self.__collect_samples(next_obs, next_done, weight_support)

        # Compute advantage on collected samples
        self.returns, self.advantages = self.__compute_advantages(next_obs, next_done, w)

        # Update neural networks from batch
        self.update()

        # Logging
        #print("SPS:", int(self.global_step / (time.time() - start_time)))
        if self.log:
            print(f"Worker {self.id} - Global step: {self.global_step}")
            wandb.log(
                {"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step},
            )
            
    def save_model(self):
        """Save MOPPO model.
        """
        saved_params = {}
        
        # Save network state dict
        saved_params["networks_state_dict"] = self.networks.state_dict()
        
        # Save optimizer state dict
        saved_params["optimizer_state_dict"] = self.optimizer.state_dict()
        
        # Save weights (equivalent to weight_support M in Q-net)
        saved_params["moppo_weights"] = self.np_weights
        return saved_params
    
    def load_model(self, params):
        """Load MOPPO model.
        """        
        # Load network state dict
        self.networks.load_state_dict(params["networks_state_dict"])
        
        # Load optimizer state dict
        self.optimizer.load_state_dict(params["optimizer_state_dict"])
        
        # Load weights
        self.change_weights(params["moppo_weights"])

    def get_action_mask(self, obs):
        """
        Get action masks for single or batched observations.
        """
        env = self.envs.envs[0].unwrapped

        # Convert to numpy if torch tensor
        if isinstance(obs, th.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.array(obs)

        # Detect single observation (1D) or batch (2D)
        is_single = obs_np.ndim == 1

        if hasattr(env, 'get_action_mask'):
            if is_single:
                # Single observation
                mask = env.get_action_mask(obs_np)
            else:
                # Batched observations — call env.get_action_mask for each one
                mask = np.stack([env.get_action_mask(o) for o in obs_np])
        else:
            # Default: all actions valid
            if is_single:
                mask = np.ones(env.action_space.n, dtype=np.float32)
            else:
                mask = np.ones((obs_np.shape[0], env.action_space.n), dtype=np.float32)

        # Convert to torch bool tensor
        return th.tensor(mask, dtype=th.bool)
