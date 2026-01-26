"""GPI-LS MOPPO algorithm."""

import os
import random
from itertools import chain
from typing import Callable, List, Optional, Union
import math

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb

import time
import cProfile
import pstats
import io

from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.utils import unique_tol
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from gpils_moppo.moppo_discrete import MOPPONet, MOPPO

class GPILS_MOPPO(MOPolicy, MOAgent):
    """GPI-LS Algorithm with Multi-Objective PPO.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 1000,  # ignored if tau != 1.0
        buffer_size: int = int(1e6),
        net_arch: List = [256, 256, 256, 256],
        num_nets: int = 2,
        batch_size: int = 128,
        timesteps_per_iter: int = 2048,
        learning_starts: int = 100,
        gradient_updates: int = 20,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = None,
        use_gpi: bool = True,
        drop_rate: float = 0.01,
        layer_norm: bool = True,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GPI-PD",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """Initialize the GPI-PD algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            tau: The soft update coefficient.
            target_net_update_freq: The target network update frequency.
            buffer_size: The size of the replay buffer.
            net_arch: The network architecture.
            num_nets: The number of networks.
            batch_size: The batch size.
            timesteps_per_iter: Timesteps per iteration of MOPPO training
            learning_starts: The number of steps before learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor.
            max_grad_norm: The maximum gradient norm.
            use_gpi: Whether to use GPI.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
            project_name: The name of the project.
            experiment_name: The name of the experiment.
            wandb_entity: The name of the wandb entity.
            log: Whether to log.
            seed: The seed for random number generators.
            device: The device to use.
        """
        single_env = env.envs[0]
        MOAgent.__init__(self, single_env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_gpi = use_gpi
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm

        self.network = MOPPONet(
                self.observation_shape,
                self.action_space.n,
                self.reward_dim,
                self.net_arch,
            ).to(self.device)

        self.steps_per_iteration = timesteps_per_iter

        # logging
        self.log = log
        weights = equally_spaced_weights(self.reward_dim, n=1)
        self.agent = MOPPO(
                1,
                self.network,
                weights[0],
                env,
                log=self.log,
                gamma=self.gamma,
                device=self.device,
                seed=self.seed,
                steps_per_iteration=self.steps_per_iteration,
                learning_rate=self.learning_rate,
            )

        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Return the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None):
        """Save model.
        
        Args:
            save_dir: Directory to save weights
            filename: Filename for the saved model
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        saved_params = self.agent.save_model()
        
        saved_params["M"] = self.weight_support

        filename = self.experiment_name if filename is None else filename
        os.makedirs(save_dir + "/" + self.full_experiment_name, exist_ok = True)
        th.save(saved_params, save_dir + "/" + self.full_experiment_name + "/" + filename + ".tar")
        
        print(f"Model saved to {save_dir}/{self.full_experiment_name}/{filename}.tar")
    
    def load(self, path):
        """Load model.
        
        Args:
            path: Path to the saved model file
        """
        params = th.load(path, map_location=self.device, weights_only = False)
        
        # Load network state dict
        self.agent.load_model(params)

        self.weight_support = params["M"]
                        
        print(f"Model loaded from {path}")

    @th.no_grad()
    def gpi_action(self, obs: th.Tensor, w: th.Tensor, return_policy_index=False, include_w=False):
        """Select an action using GPI."""
        if include_w:
            M = th.stack(self.weight_support + [w])
        else:
            M = th.stack(self.weight_support)

        obs_m = obs.repeat(M.size(0), *(1 for _ in range(obs.dim())))

        logits = self.agent.get_action_logits(obs_m, M)
        weighted_logits = th.einsum("r,bar->ba", w, logits)  # q(s,a,w_i) = q(s,a,w_i) . w
        distr = Categorical(logits=weighted_logits)
        action_probs = distr.probs

        max_prob, a = th.max(action_probs, dim=1)
        policy_index = th.argmax(max_prob)  # max_i max_a q(s,a,w_i)
        action = a[policy_index].detach().item()

        if return_policy_index:
            return action, policy_index.item()
        return action

    @th.no_grad()
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        """Select an action for the given obs and weight vector."""
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        #for q_net in self.q_nets:
        #    q_net.eval()
        if self.use_gpi:
            action = self.gpi_action(obs, w, include_w=False)
        else:
            action = self.max_action(obs, w)
        #for q_net in self.q_nets:
        #    q_net.train()
        return action

    def _act(self, obs: th.Tensor, w: th.Tensor) -> int:
        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.use_gpi:
                action, policy_index = self.gpi_action(obs, w, return_policy_index=True)
                self.police_indices.append(policy_index)
                return action
            else:
                return self.max_action(obs, w)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Select the greedy action."""
        logits = self.agent.get_action_logits(obs, w)
        weighted_logits = th.einsum("r,bar->ba", w, logits)
        distr = Categorical(logits=weighted_logits)
        action_probs = distr.probs

        max_act = th.argmax(action_probs, dim=1)
        return max_act.detach().item()

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeats = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeats]

    def train(
        self,
        total_timesteps: int,
        eval_env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        timesteps_per_weight_iter: int = 10000,
        weight_selection_algo: str = "gpi-ls",
        eval_freq: int = 1000,
        eval_mo_freq: int = 10000,
        checkpoints: bool = True,
    ):
        """Train agent.

        Args:
            total_timesteps (int): Number of timesteps to train for.
            eval_env (gym.Env): Environment to evaluate on.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front if known.
            num_eval_weights_for_front: Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            timesteps_per_iter (int): Number of timesteps to train for per iteration.
            weight_selection_algo (str): Weight selection algorithm to use.
            eval_freq (int): Number of timesteps between evaluations.
            eval_mo_freq (int): Number of timesteps between multi-objective evaluations.
            checkpoints (bool): Whether to save checkpoints.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "timesteps_per_iter": self.steps_per_iteration,
                    "weight_selection_algo": weight_selection_algo,
                    "eval_freq": eval_freq,
                    "eval_mo_freq": eval_mo_freq,
                }
            )

        pr = cProfile.Profile()
        pr.enable()

        train_start_time = time.time()
        outer_iter = 1 if total_timesteps <= timesteps_per_weight_iter else math.ceil(total_timesteps / timesteps_per_weight_iter)
        inner_iter = 1 if timesteps_per_weight_iter <= self.steps_per_iteration else math.ceil(timesteps_per_weight_iter / self.steps_per_iteration)
        max_iter = outer_iter * inner_iter
        eval_iter_freq = 1 if eval_mo_freq <= self.steps_per_iteration else math.ceil(eval_mo_freq / self.steps_per_iteration)

        linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=0.0 if weight_selection_algo == "ols" else None)

        weight_history = []

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        current_iter = 1
        for iter in range(1, outer_iter + 1):
            start_time = time.time()
            print("Iteration: {}\n".format(iter))
            if weight_selection_algo == "ols" or weight_selection_algo == "gpi-ls":
                if weight_selection_algo == "gpi-ls":
                    self.set_weight_support(linear_support.get_weight_support())
                    use_gpi = self.use_gpi
                    self.use_gpi = True
                    w = linear_support.next_weight(
                        algo="gpi-ls", gpi_agent=self, env=eval_env, rep_eval=num_eval_episodes_for_front
                    )
                    self.use_gpi = use_gpi                         
                    print("--Next Weight ", w)    
                else:
                    w = linear_support.next_weight(algo="ols")
                if w is None:
                    break
            else:
                raise ValueError(f"Unknown algorithm {weight_selection_algo}.")   
            print("--- Get next weight duration: %s seconds --- \n\n" % (time.time() - start_time))
            weight_history.append(w)
            if weight_selection_algo == "gpi-ls":
                M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
            elif weight_selection_algo == "ols":
                M = linear_support.get_weight_support() + [w]
            else:
                M = None

            unique_M = unique_tol(M)  # remove duplicates
            self.set_weight_support(unique_M)
            
            # Train PPO agent
            start_time_ppo = time.time() 
            for i in range(inner_iter):
                start_time_train = time.time()
                self.agent.train(start_time_train, current_iter, max_iter, weight = w, weight_support = unique_M)
                current_iter += 1
            self.global_step = self.agent.global_step
            print("--- PPO training duration: %s seconds --- \n\n" % (time.time() - start_time_ppo))

            start_time_sol = time.time() 
            if weight_selection_algo == "ols":
                value = policy_evaluation_mo(self, eval_env, w, rep=num_eval_episodes_for_front)[3]
                linear_support.add_solution(value, w)
            elif weight_selection_algo == "gpi-ls":
                # TODO
                #for wcw in M:
                for wcw in unique_M:
                    n_value = policy_evaluation_mo(self, eval_env, wcw, rep=num_eval_episodes_for_front)[3]
                    linear_support.add_solution(n_value, wcw)
            print("--- add solution duration: %s seconds --- \n\n" % (time.time() - start_time_sol))
            print("current ccs ", linear_support.ccs)
            print("current ccs weights ", linear_support.weight_support)
            
            if self.log and iter % eval_iter_freq == 0:
                # Evaluation
                gpi_returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[2] for ew in eval_weights
                ]
                print("gpi return len", len(gpi_returns_test_tasks))
                log_all_multi_policy_metrics(
                    current_front=gpi_returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                    eval_weights = eval_weights,
                )
                # This is the EU computed in the paper
                mean_gpi_returns_test_tasks = np.mean(
                    [np.dot(ew, q) for ew, q in zip(eval_weights, gpi_returns_test_tasks)], axis=0
                )
                wandb.log({"eval/Mean Utility - GPI": mean_gpi_returns_test_tasks, "iteration": iter})
            
            if checkpoints:
                self.save(filename=f"GPI-PD {weight_selection_algo} iter={iter}")
            print("--- Iteration end: %s seconds --- \n\n" % (time.time() - start_time))

        print("--- Training End: %s seconds --- \n\n" % (time.time() - train_start_time))
        print("Weight History", weight_history)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        print(ps.print_stats("morl_baselines"))

        if self.log:
            profiling_summary = s.getvalue()
            wandb.log({"profiling_summary": wandb.Html(f"<pre>{profiling_summary}</pre>")})

            self.close_wandb()

    def update(self, *args, **kwargs):
        """Dummy update function (not used for PPO-based GPILS)."""
        pass