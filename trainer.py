from datetime import datetime
import pickle as pkl
import torch
from policies import risk_seeking_policy
import time
from BTS_Transformer_model import *
import os
import numpy as np
from expression_tree import simplify_equation, ExpressionTree
import warnings
import copy
from enums import *
import traceback
import math
import random

CUDA = torch.device('cuda')
CPU = torch.device('cpu')


class CADSR:
    def __init__(self, x_dim, **kwargs):
        """
        Initialization function for CADSR. The all parameters other than feature dimensionality are already have default
        parameters that should work well for most problems.
        :param x_dim: Number of features in the dataset, ie the dimensionality of the dataset
        :param two_children_funcs: All binary operators being used. All of these operators are assumed to be native
        operators of the form x_1 (operator) x_2
        :param one_children_funcs: All unary operators being used. All of these operators are assumed to be supported by
        numpy. The method expects just the function call to be given such as 'log', and the operators should work by
        calling np.'operator'(x_1)
        :param variables: A list of variables to use. If this is none the list will be auto-generated to contain all
        input features, and 1. Please use the opt_const flag for controlling if an optimizable constant token is desired
        in the variable list.
        :param max_depth: The maximum number of tokens in the expression tree. Making this number small limits the depths
        of the trees, but decreases the search space.
        :param lr: The learning rate for backprop. Lowering this number will slow the rate at which the model alters the
        learned prior distributions. There is a trade off here between search time and quality of the search.
        :param oversampling: A scalar for the number of equations in each sample to increase the chance that all equations
        sampled in a batch are unique. Increase this number will increase the number of unique equations at the cost of runtime.
        Increasing this number above 5 is unlikely, and the number should never go bellow 1.
        :param weight_decay: A form of regularization for the model, can be ignored by most users.
        :param positional_encoding: A testing parameter. Users should just use the default PE of TwoDPE.
        :param expression_tree_device: The device the expression trees will be stored on. Can be either CPU or CUDA, but
        fastest performance is on CUDA.
        :param transformer_device: The device the model will be on. Should almost always be on CUDA as it has a 10x
        faster run time atleast. NOTE: HAVING THE EXPRESSION TREES AND MODEL ON DIFFERENCE DEVICES SHOULD BE SUPPORTED,
        BUT HASN'T BEEN TESTED RECENTLY AND COULD CAUSE UNSEEN BUGS.
        :param opt_const: Flag for if an optimizable constant token should be included in the variable list.
        :param num_heads: Number of heads in the transformer. One should be sufficient. Must divide the embedding space without
        remainder.
        :param dim_feedforward: Hidden dimensionality of the transformer. Hidden dimension default should be sufficient.
        :param embedding_dim: The dimensionality of the token space.
        :param encoder_layers: Number of encoder layers in the transformer. One should be sufficient.
        :param decoder_layers: Number of decoder layers in the transformer. One should be sufficient.
        :param entropy_coef: A weight for the entropy loss. This value is an additional way to control exploration. The
        default value should suffice. If the max depth is increased, decreasing this value might be helpful, and the
        opposite if max depth is decreased.
        :param save_ratio: This controls the Long-Term Short-Term Queue Policy, and can improve performance. Test on values of 0.0
        and 1.0 can be a good place to start when iterating on a specific dataset. If this value is set to 1.0 the trainer
        will retain the over top (epsilon * batch_size) number of equations as targets for the model to use during
        backpropagation in addition to the top epsilon % of equations sampled during the given epoch.
        :param reward_function: List of reward functions are located in enums.RewardFunctions. The two best ones are BIC
        and NMSE.
        :param risk_epsilon: Controls the ratio of equations that the model is trained on from a given batch. For example,
        using the default value the model is trained on the top 5% of equations from each given batch. Decreasing this
        number can lead to more risky exploration, while increasing this number makes the model make more consistent
        safer predictions.
        :param policy: This enum controls a part of the policy. For NMSE this should be set to Linear. For
        BIC this should be set to Uniform, but additional experimentation can be done with setting it to False as well.
        :param bic_scaler: This controls 'cost' of the size of the expression tree compared to the data. If you want
        larger expression trees, while still using a BIC reward function, decrease this value. If you want simpler
        expression trees increase this value.
        :param optimizer: The kind of optimizer used for optimizing the equations. The two choices are 'lm' and 'L-BFGS'.
        Typically, 'lm' is best but will increase runtime when compared to 'L-BFGS'.
        :param max_cpu_count: The maximum number of cpus used to optimize the equations, currently this will only improve
        performance if the code is ran in a jupyter notebook. If this is not ran in a notebook set it to 1.
        :param suppress_warnings: Set to true to clean up printed outputs. If false, you will see all warnings, and/or
        errors that when evaluating every expression tree.
        :param seed: RNG seed, if RNG is set prior to running the method
        """

        default_parameters = {
            "two_children_funcs": [],
            "one_children_funcs": [],
            "variables": [],
            "expression_tree_device": "cuda",
            "seed": 123,
            "model_parameters": {
                "max_depth": 32,
                "num_heads": 1,
                "dim_feedforward": 2048,
                "encoder_layers": 0,
                "parent_sibling_info_in_encoder": False,
                "decoder_layers": 1,
                "oversampling": 3,
                "opt_const": True,
                "use_dct": True,
                "embedding_dim": 10,
                "dct_dim": 8,
                "max_num_const": 10,
                "pe": "TwoDPE",
                "device": "cuda"
            },
            "optimizer_parameters": {
                "lr": 1E-4,
                "weight_decay": 0.0
            },
            "reward_function": "NMSE",
            "policy": "Basic",
            "policy_scaling": "Linear",
            "sampling_method": "Autoregressive",
            "time_sampling_dist": "Uniform",
            "resample_times": False,
            "risk_epsilon": 0.05,
            "save_ratio": 0.0,
            "bic_scaler": 1.0,
            "entropy_coef": 0.005,
            "equation_optimizer": "lm",
            "beta": 0.1,
            "epsilon_clip": 0.2,
            "epochs_per_ref": 1,
            "steps_per_sample": 1,
            "suppress_warnings": True,
            "max_cpu_count": 1,
            "save_loc": "\\..\\run_data\\",
            "base_name": "CADSR",
            "save_timings": True,
            "save_epoch_info": True,
            "save_eq_dict": False
        }

        for key, value in default_parameters.items():
            if value is dict:
                for emb_key, emb_value in value:
                    if emb_key not in kwargs[key].keys():
                        kwargs[key][emb_key] = emb_value
            if key not in kwargs.keys():
                kwargs[key] = value

        # Trying to set values to enums
        try:
            kwargs["model_parameters"]["pe"] = PositionalEncodings(kwargs["model_parameters"]["pe"])
            kwargs["sampling_method"] = SamplingMethod(kwargs["sampling_method"])
            kwargs["reward_function"] = RewardFunctions(kwargs["reward_function"])
            kwargs["policy"] = Policies(kwargs["policy"])
            kwargs["policy_scaling"] = RiskSeekingPolicyScaling(kwargs["policy_scaling"])
        except TypeError:
            raise TypeError("Failed to convert string to enum")

        # Trying to find the devices
        kwargs["expression_tree_device"] = torch.device(kwargs["expression_tree_device"])
        kwargs["model_parameters"]["device"] = torch.device(kwargs["model_parameters"]["device"])

        torch.set_default_dtype(torch.float32)

        if kwargs["suppress_warnings"]:
            warnings.filterwarnings("ignore")
            np.seterr(all='ignore')

        if len(kwargs["one_children_funcs"]) == 0:
            kwargs["one_children_funcs"] = ["log", "sin", "cos", "sqrt", "exp"]
        if len(kwargs["two_children_funcs"]) == 0:
            kwargs["two_children_funcs"] = ["+", "-", "*", "/", "**"]

        kwargs["one_children_funcs"] = ["np." + s for s in kwargs["one_children_funcs"]]
        if len(kwargs["variables"]) == 0:
            kwargs["variables"] = [f"x[{i}]" for i in range(x_dim)]
            kwargs["variables"] += ["1"]
        if kwargs["model_parameters"]["opt_const"]:
            kwargs["variables"] += ["const"]

        model_map = {
            SamplingMethod.Autoregressive: AutoregressiveModel
        }

        model_class = model_map.get(kwargs["sampling_method"])
        if model_class is None:
            raise TypeError("Failed to specify a valid sampling method.")

        self.model = model_class(
            kwargs["two_children_funcs"],
            kwargs["one_children_funcs"],
            kwargs["variables"],
            **(kwargs["model_parameters"])
        )

        self.model.to(self.model.device)

        optimizer_parameters = kwargs["optimizer_parameters"]
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=optimizer_parameters["lr"], weight_decay=optimizer_parameters["weight_decay"])

        self.epoch_info = None
        self.training_info = None
        self.timer_dictionary = None
        self.eq_dict = None
        self.ref_model = copy.deepcopy(self.model) if kwargs["policy"] == Policies.GRPO else None
        self.use_lm = (kwargs["equation_optimizer"] == "lm")
        self.reward_function = kwargs["reward_function"]
        self.policy_scaling = kwargs["policy_scaling"]
        self.policy = kwargs["policy"]
        self.time_sampling_dist = kwargs["time_sampling_dist"]
        self.entropy_coef = kwargs["entropy_coef"]
        self.save_ratio = kwargs["save_ratio"]
        self.bic_scaler = kwargs["bic_scaler"]
        self.steps_per_sample = kwargs["steps_per_sample"]
        self.epochs_per_ref = kwargs["epochs_per_ref"]
        self.beta = kwargs["beta"]
        self.epsilon_clip = kwargs["epsilon_clip"]
        self.sampling_method = kwargs["sampling_method"]
        self.risk_epsilon = kwargs["risk_epsilon"]
        self.resample_ts = kwargs["resample_times"]
        self.max_cpu_count = kwargs["max_cpu_count"]
        ps_info = True

        self.best_trees = ExpressionTree(n=0, two_children_funcs=self.model.two_children_funcs,
                                         one_children_funcs=self.model.one_children_funcs, variables=self.model.variables,
                                         max_depth=self.model.max_depth, max_num_const=self.model.max_num_const,
                                         time_steps=1, max_layers_steps=1,
                                         opt_const=self.model.opt_const, device=self.model.device, ps_info=ps_info)
        self.run_info = {
            "Hyper-parameters": kwargs,
            "Best Function": {"Equation": "", "Constants": [], "Loss": -torch.inf, "R2": 0.0},
            "Training Cycle": []
        }
        self.best_func = self.run_info["Best Function"]

        count = 0
        file_name = f"{kwargs['base_name']}_{count}.pkl"
        self.save_loc = kwargs["save_loc"]
        # Check if the file already exists, incrementing count if necessary
        while os.path.exists(os.path.curdir + self.save_loc + file_name):
            count += 1
            file_name = f"{kwargs['base_name']}_{count}.pkl"
        self.save_name = file_name
        self.save_epoch_info = kwargs["save_epoch_info"]
        self.save_timings = kwargs["save_timings"]
        self.save_eq_dict = kwargs["save_eq_dict"]
        print(f"Training Info Saving At {self.save_name}")

 

    def train(self, x, y, epochs=2000, batch=1000, print_counts=10, save_timings=True, save_epoch_info=True, save_eq_dict=True,
              verbose=True,  max_runtime=1E9, termination_acc=None):
        """
        This method controls one training session. Multiple calls to train can be made, but typically just one call will
        be necessary. The most common variables to change will be decreasing epochs, changing reward functions, and
        save ratio.
        :param termination_acc: The loss score needed for early termination
        :param print_counts: The number of times information about the training process is printed
        :param x: The input features, expected in the form [Feature dim, Batch dim]. Can be either a numpy array or a
        tensor.
        :param y: The targets, expected in the form [Batch dim]. Can be either a numpy array of a tensor.
        :param epochs: Number of epochs for training. Default should be more than enough, and most problems could
        potential use 500 for initial testing. For extremely complex problems, and if the transformer architecture
        is increased a larger number of epochs can be used.
        :param batch: The number of equations sampled in each epoch. Increasing this number will increase the models
        ability to effectively explore a given epoch. For larger more complex problems this number could be increased,
        but will cause an increase in runtime.
        :param save_timings: If true timings for the training are included in the save dictionary.
        :param save_epoch_info: If true information about each epoch are saved in the save dictionary.
        :param save_eq_dict: If true all of the equations found are saved in a dictionary with performance data.
        NOTE this will take up a lot space depending upon the number equations discovered. For example with 2000 epochs
        expect the save file to be larger than 6 GB.
        :param verbose: If true will print out information about the run every so often.
        :param max_runtime: This is the maximum number of seconds the method can run for before terminating. On the epoch
        when the threshold is pasted the method will wrap up and save, which can take additional seconds - minutes.
        :return: A dictionary containing the information about the best equation discovered.
        """
        print_index = int(epochs / print_counts) if int(epochs / print_counts) > 1 else 1

        # initialize
        if torch.is_tensor(x):
            x = x.numpy()
        if torch.is_tensor(y):
            y = y.numpy()

        start_time = time.perf_counter()
        self.epoch_info = {"Loss": [], "Policy Loss": [], "Entropy Loss": [], "KL Loss": [], "Epoch Time": [], "Best Reward": [],
                           "Median Reward": [], "Baseline Reward": [], "Best Function": [], "Rewards": [], "Depths": [],
                           "Expression Losses": [], "Full Entropy": [], "Node Counts": [], "New Equations": [], "Entropy" : [], "Alpha Entropy": []}
        self.timer_dictionary = {"Sample Time": [], "Sample Time In-depth": [], "Opt Time": [], "Reward": [], "Prediction": [], "Epoch Time": []}
        self.training_info = {
            "parameters": {
                "epochs": epochs,
                "batch": batch
            },
            "Training Data": (x, y),
            "Total Entropy": torch.zeros((self.model.max_depth, self.model.library_size), device=self.model.device),
            "Total Alpha Entropy": torch.zeros((self.model.max_depth, self.model.library_size), device=self.model.device),
        }
        self.eq_dict = {}

        for i in range(epochs):

            if i % self.epochs_per_ref == 0:
                self.ref_model = copy.deepcopy(self.model)
            try:
                losses, policy_losses, entropy_losses, kl_losses, policy_info, trees = self.training_epoch(i, x, y, batch)
            except Exception as e:
                print(f"Exception Type: {type(e).__name__}")
                print(f"Message: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                break

            self.epoch_info["Loss"] += losses
            self.epoch_info["Policy Loss"] += policy_losses
            self.epoch_info["Entropy Loss"] += entropy_losses
            self.epoch_info["KL Loss"] += kl_losses
            self.epoch_info["Best Reward"].append(policy_info[0])
            self.epoch_info["Median Reward"].append(policy_info[1])
            self.epoch_info["Baseline Reward"].append(policy_info[2])
            self.epoch_info["Best Function"].append(trees.equation_string()[0])

            # Print Info
            if (i + 1) % print_index == 0:
                self.print_info(i, start_time, policy_losses, entropy_losses, losses, verbose)
                self.save_results()

            if max_runtime < time.perf_counter() - start_time:
                print("Exceeded Max Runtime")
                self.print_info(i, start_time, policy_losses, entropy_losses, losses, verbose)
                break

            if termination_acc is not None:
                if self.best_func["Loss"] > termination_acc:
                    print("Accuracy Reched")
                    self.print_info(i, start_time, policy_losses, entropy_losses, losses, verbose)
                    break
            
        # Trying to simplify final function
        try:
            self.best_func["Simplified Equation"] = simplify_equation(self.best_func["Equation"], len(self.best_func["Constants"]))
        except:
            print("Failed to simplify final function")

        self.save_results()

        return self.best_func["Equation"]

    def training_epoch(self, i, x, y, batch):
        # Sample from the model
        cycle_time = time.perf_counter()
        a = time.perf_counter()
        trees, times = self.model.sample(batch, self.model.device)
        sample_dist = trees.get_labels().float().mean(dim=0)
        self.training_info["Total Entropy"] += sample_dist
        self.epoch_info["Entropy"].append(-torch.sum(sample_dist * torch.log(sample_dist + 1e-8)).item())
        node_counts = trees.get_node_counts().to(CPU)
        self.epoch_info["Node Counts"].append(node_counts)
        self.epoch_info["Depths"].append(trees.positions.max(dim=1)[0][:, 0].tolist())
        self.timer_dictionary["Sample Time In-depth"].append(times)
        self.timer_dictionary["Sample Time"].append(time.perf_counter() - a)
        # optimize sampled functions
        a = time.perf_counter()
        trees.max_cpu_count = self.max_cpu_count
        trees.opt(x, y, self.reward_function, self.use_lm, self.bic_scaler)
        r_2s = trees.calc_r2s(x, y)
        new_equations_counter = 0
        for k, eq in enumerate(trees.equation_string()):
            if eq not in self.eq_dict.keys():
                self.eq_dict[eq] = {"Reward": trees.rewards[k], "Node Count": node_counts[k], "R2": r_2s[k], "Epoch": i}
                new_equations_counter += 1
        self.epoch_info["New Equations"].append(new_equations_counter)
        self.epoch_info["Rewards"].append(trees.rewards)
        self.timer_dictionary["Opt Time"].append(time.perf_counter() - a)

        # Add a policy that evaluates only the top x%
        a = time.perf_counter()
        trees, baseline, policy_info = risk_seeking_policy(trees, self.risk_epsilon)

        self.timer_dictionary["Reward"].append(time.perf_counter() - a)
        sample_dist = trees.get_labels().float().mean(dim=0)
        self.training_info["Total Alpha Entropy"] += sample_dist
        self.epoch_info["Alpha Entropy"].append(-torch.sum(sample_dist * torch.log(sample_dist + 1e-8)).item())

        a = time.perf_counter()
        # Checks to see if a new best tree has been found
        best_ind = np.nanargmax(trees.rewards)
        if trees.rewards[best_ind] > self.best_func["Loss"]:
            self.best_func["Equation"] = trees.equation_string()[best_ind]
            self.best_func["Constants"] = trees.constants[best_ind].tolist()
            self.best_func["Loss"] = trees.rewards[best_ind]
            self.best_func["R2"] = self.eq_dict[trees.equation_string()[best_ind]]["R2"]
            if self.reward_function == RewardFunctions.BIC:
                self.best_func["Noise"] = trees.noise[best_ind].tolist()

        if self.save_ratio == 0.0:
            self.best_trees = trees
        else:
            self.best_trees.join(trees)

        losses, neglogpes, entropies, kl_losses = self.take_step(self.best_trees, baseline)

        if self.save_ratio != 0.0:
            epoch_save_ratio = self.save_ratio * int(self.risk_epsilon * batch) / self.best_trees.n
            self.best_trees,  _, _ = risk_seeking_policy(self.best_trees, epoch_save_ratio)

        self.timer_dictionary["Prediction"].append(time.perf_counter() - a)
        self.timer_dictionary["Epoch Time"].append(time.perf_counter() - cycle_time)

        return losses, neglogpes, entropies, kl_losses, policy_info, trees

    def take_step(self, trees, baseline):
        losses, policy_losses, entropies, kl_losses = [], [], [], []
        old_logits = None
        old_model = copy.deepcopy(self.model)

        for i in range(self.steps_per_sample):
            
            kl_loss = torch.tensor(0)
            self.opt.zero_grad()
            # make prediction
            logits, targets = self.get_logits_and_targets(self.model, trees)

            if old_logits is None:
                old_logits = torch.ones(logits.shape, device=self.model.device)

            advantage_matrix = self.calc_advantage(trees, baseline)

            if self.policy == Policies.Basic:
                policy_loss = self.calc_NLL(trees, logits, advantage_matrix, targets)

            elif self.policy == Policies.PPO:  # I don't think this is right yet
                policy_loss = self.calc_PPO(logits, old_logits, advantage_matrix, targets)

            else:
                ref_logits, ref_targets = self.get_logits_and_targets(self.ref_model, trees)
                policy_loss = self.calc_PPO(logits, old_logits, advantage_matrix, targets)
                kl_loss = self.calc_KL(logits, ref_logits)
                policy_loss += kl_loss

            if self.policy is Policies.Basic:
                entropy = self.entropy_coef * torch.mean(torch.sum(logits.float() * torch.log(logits.float()), dim=2))
            else:
                entropy = self.entropy_coef * torch.mean(torch.sum(logits.float() * torch.log(logits.float()), dim=2)).exp()

            loss = policy_loss + entropy
            loss.backward()
            self.opt.step()

            old_logits = logits.detach()

            losses.append(loss.detach().item())
            policy_losses.append(policy_loss.detach().item())
            entropies.append(entropy.detach().item())
            kl_losses.append(kl_loss.detach().item())

            if loss.isnan():
                self.print_info()
                print("Loss is NaN")

        return losses, policy_losses, entropies, kl_losses

    def get_logits_and_targets(self, model, trees):

        if self.sampling_method == SamplingMethod.Autoregressive:
            ps_information = trees.get_inputs().float().to(model.device)
            positions = trees.get_positions().float().to(model.device)
            targets = trees.get_labels().float().to(model.device)
            logits = model(targets, ps_information, positions)

        return logits, targets

    def calc_NLL(self, trees, logits, advantage_matrix, labels):
        mask = (torch.arange(trees.max_depth).unsqueeze(0).repeat(trees.n, 1).to(trees.device) < trees.node_counts.unsqueeze(1).repeat(1, trees.max_depth)).float()
        step_neglogp = torch.sum(-torch.sum(torch.log(logits) * labels, dim=2) * mask, dim=1)
        neglogp = torch.mean(advantage_matrix * step_neglogp)
        return neglogp

    def calc_PPO(self, logits, old_logits, advantage, targets):
        step_logp = -torch.sum(torch.log(logits) * targets, dim=2)
        step_logp_old = -torch.sum(torch.log(old_logits) * targets, dim=2)
        ratio = (-step_logp + step_logp_old.detach()).exp()
        policy_loss = (-torch.min(
            torch.sum(ratio * advantage, dim=1),
            torch.sum(torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage, dim=1)
        )).mean()
        return policy_loss

    def calc_KL(self, new_logits, ref_logits):
        d_kl = torch.functional.F.kl_div(new_logits, ref_logits, reduction='mean')
        d_kl = torch.clamp(d_kl, min=-10, max=10)
        return - self.beta * d_kl

    def calc_advantage(self, trees, baseline):
        rewards = torch.tensor(trees.rewards, device=trees.device, dtype=torch.float32)
        if self.policy_scaling == RiskSeekingPolicyScaling.Uniform:
            advantage = torch.ones(trees.n, device=trees.device)/12.5
        elif self.policy_scaling == RiskSeekingPolicyScaling.Linear:
            if self.reward_function == RewardFunctions.BIC:
                advantage = torch.range(len(trees.rewards), 1, -1, device=trees.device) / (5 * len(trees.rewards))
            else:
                advantage = rewards - baseline
        elif self.policy_scaling == RiskSeekingPolicyScaling.RunningMean:
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1E-5)
        else:
            advantage = (-torch.range(1, len(trees.rewards), 1, device=trees.device)/5 - 1).exp()
        lengths = trees.get_node_counts().unsqueeze(1) * torch.ones(trees.n, trees.max_depth, device=trees.device) - torch.range(0, trees.max_depth - 1, device=trees.device).unsqueeze(0)
        lengths[lengths < 0] = 0
        scaled_advantage = torch.pow(1.0, lengths) * advantage.unsqueeze(1)
        return scaled_advantage

    def print_info(self, i, start_time, policy_losses, entropy_losses, losses, verbose):
        if not verbose:
            return
        print(f"Reward: {self.best_func['Loss']}, R2: {self.best_func['R2']}, Best Equation: {self.best_func['Equation']}")
        print(f"Policy Loss: {np.mean(policy_losses)}, Entropy: {np.mean(entropy_losses)}")
        print(f"Epoch: {i}, Loss: {np.mean(losses)}")
        print(f"Run time: {time.perf_counter() - start_time: .3f}, Avg Epoch Time: {(time.perf_counter() - start_time) / (i + 1) : .3f}")
        # print(self.best_trees.rewards[0], self.best_trees.equation_string()[0])

    def add_test_info(self, x, y):
        """
        This method allows you to add the test information, so that it is saved along side the final model
        made for simpler loading and evaluation.
        :param x:  The input features, expected in the form [Feature dim, Batch dim]. Can be either a numpy array or a
        tensor.
        :param y: The targets, expected in the form [Batch dim]. Can be either a numpy array of a tensor.
        """
        if torch.is_tensor(x):
            x = x.numpy()
        if torch.is_tensor(y):
            y = y.numpy()
        self.run_info["Test Data"] = (x, y)
        self.save_results()

    def save_model(self, loc):
        torch.save(self.model.state_dict(), loc)

    def load_model(self, loc):
        self.model.load_state_dict(torch.load(loc))

    def save_results(self, save_timings=None, save_epoch_info=None, save_eq_dict=None):
        """
        This method saves the results, and any information from training. Please test this method before you run your
        experiments to make sure the file is saving in the correct location as there are differences between how this
        method functions based on operating systems.
        :param base_name: The number of the file, pkl will be added, along with a count incase another file with this
        name already exists.
        :param loc: the folder when the file will be located. If this is the local space just leave it empty
        """
        if save_timings is None:
            save_timings = self.save_timings
        if save_epoch_info is None:
            save_epoch_info = self.save_epoch_info
        if save_eq_dict is None:
            save_eq_dict = self.save_eq_dict

        # Saving info
        i = len(self.epoch_info["Loss"])
        self.run_info["Best Function"] = self.best_func
        sample_dist_entropy = self.training_info["Total Entropy"] / i
        self.training_info["Total Entropy"] = -torch.sum(sample_dist_entropy *  torch.log(sample_dist_entropy + 1e-8)).item()
        sample_dist_alpha = self.training_info["Total Alpha Entropy"] / i
        self.training_info["Total Alpha Entropy"] = -torch.sum(sample_dist_alpha *  torch.log(sample_dist_alpha + 1e-8)).item()

        if save_timings:
            self.training_info["Timings"] = self.timer_dictionary
        if save_epoch_info:
            self.training_info["Iteration Info"] = self.epoch_info
        if save_eq_dict:
            self.training_info["All Equations Tested"] = self.eq_dict

        self.run_info["Training Cycle"].append(self.training_info)

        with open(os.getcwd() + self.save_loc + self.save_name, 'wb') as file:
            pkl.dump(self.run_info, file)

        self.training_info["Total Entropy"] = i * sample_dist_entropy
        self.training_info["Total Alpha Entropy"] = i * sample_dist_alpha
        # print(f"Dictionary has been saved as JSON in {self.save_name}")
