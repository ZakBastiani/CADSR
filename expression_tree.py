import os
import torch
from multiprocessing import Pool
from torch.nn import functional as F
from torch.distributions import Categorical
from expression_tree_helpers import *

np.seterr(divide='ignore', invalid='ignore')


class ExpressionTree:
    def __init__(self, n, two_children_funcs, one_children_funcs, variables, max_depth, max_num_const=10,
                 opt_const=True, max_layers_steps=1, time_steps=1, max_cpu_count=1, device=torch.device("cuda"),
                 ps_info=True):
        self.n = n
        self.max_dataset_size = 1000
        self.max_cpu_count = max_cpu_count
        self.max_depth = max_depth
        self.library = two_children_funcs + one_children_funcs + variables
        self.opt_const = opt_const
        self.library_size = len(self.library)
        self.two_children_num = len(two_children_funcs)
        self.one_children_num = len(two_children_funcs) + len(one_children_funcs)
        if ps_info:
            self.input_size = 2 * (len(self.library) + 1)
        else:
            self.input_size = self.library_size
        self.max_num_const = max_num_const
        self.device = device
        self.empty = torch.zeros((1, 1, self.input_size), device=self.device, dtype=torch.bool)
        self.constants = [np.random.rand(1)] * n
        self.noise = [np.random.rand(1)] * n
        self.rewards = np.zeros(n)
        self.equations = []
        self.inputs_backlog = torch.zeros((self.n, max_depth + 2, 2), device=self.device, dtype=torch.int32)
        self.children_locations = torch.zeros((self.n, max_depth), device=self.device, dtype=torch.int32)

        self.positions_history = torch.zeros((self.n, max_layers_steps, time_steps, max_depth, 2), device=self.device, dtype=torch.int32)

        self.bforder = torch.zeros((self.n, max_depth), device=self.device, dtype=torch.int32)
        self.has_sibling = torch.zeros((self.n, max_depth + 2), device=self.device, dtype=torch.bool)
        self.valid_nodes = torch.zeros((self.n, max_depth + 2), device=self.device, dtype=torch.bool)
        self.is_const = torch.zeros((self.n, max_depth), device=self.device, dtype=torch.bool)
        self.is_const_parent = torch.zeros((self.n, max_depth + 2), device=self.device, dtype=torch.bool)
        self.is_const_sibling = torch.zeros((self.n, max_depth + 2), device=self.device, dtype=torch.bool)
        self.positions = torch.zeros((self.n, max_depth + 2, 2), device=self.device, dtype=torch.float64)
        self.positions[:, 0, 0] = 1
        self.positions[:, 0, 1] = 0.5
        self.incremental_constant = [0] * n
        self.node_counts = torch.ones(self.n, device=device, dtype=torch.int32)
        self.rules = torch.ones((n, max_depth + 2, self.library_size), device=self.device, dtype=torch.bool)
        # Rules for the expression tree
        self.constants_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.ONF_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.one_func_or_vars_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.vars_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.constants_rule[-1] = 0.0
        for i, func_name in enumerate(one_children_funcs):
            if func_name in ["np.cos", "np.sin"]:
                self.ONF_rule[i + len(two_children_funcs)] = 0.0
        for j in range(self.two_children_num):
            self.one_func_or_vars_rule[j] = 0.0
        for j in range(self.one_children_num):
            self.vars_rule[j] = 0.0

    # Evaluates x for the expression tree
    def evaluate(self, x):
        equations = self.equation_string()
        y = []
        for i in range(self.n):
            self.incremental_constant[i] = 0
            c = self.constants[i]
            y.append(eval(equations[i]))
        return y

    def equation_string(self):
        if len(self.equations) != 0:
            return self.equations
        self.equations = []
        temp = (self.bforder * self.valid_nodes[:, :self.max_depth]).to(torch.device("cpu"))
        for n in range(self.n):
            equ = "s0"
            j = 0
            self.incremental_constant[n] = 0
            self.constants[n] = []
            for i, element in enumerate(temp[n]):
                if not self.valid_nodes[n, i]:
                    break
                if element < self.two_children_num:
                    equ = equ.replace(f"s{i}", f"(s{j + 1} {self.library[element]} s{j + 2})")
                    j += 2
                elif element < self.one_children_num:
                    equ = equ.replace(f"s{i}", f"{self.library[element]}(s{j + 1})")
                    j += 1
                elif element != self.library_size - 1:
                    equ = equ.replace(f"s{i}", f"{self.library[element]}")
                elif self.opt_const:
                    equ = equ.replace(f"s{i}", f"c[{self.incremental_constant[n]}]")
                    self.incremental_constant[n] += 1
                else:
                    equ = equ.replace(f"s{i}", f"{self.library[element]}")

            self.equations.append(equ)

            if len(self.constants[n]) < self.incremental_constant[n]:
                self.constants[n] = np.concatenate((self.constants[n], np.random.rand(self.incremental_constant[n] - len(self.constants[n]))))
                self.constants[n] = self.constants[n][:self.incremental_constant[n]]
            else:
                self.constants[n] = np.array(self.constants[n])
        return self.equations

    # Sample full trees
    def sample_full_trees(self, priors):
        for j in range(self.max_depth):
            rules = self.fetch_rules(j)
            temp = priors[:, j, :] * rules
            if temp.isnan().any():
                raise ValueError("Nans in prob")
            predicted_vals = categorical_sample(temp)
            self.add(predicted_vals.int(), j)

    # Sample new tree with the same structure
    def sample_full_trees_same_struct(self, priors, previous_structure):
        for j in range(self.max_depth):
            node_class = self.get_node_class(previous_structure, j)
            temp = priors[:, j, :] * node_class
            if temp.isnan().any():
                raise ValueError("Nans in prob")
            predicted_vals = categorical_sample(temp)
            self.add(predicted_vals.int(), j)

    def get_node_class(self, prev_struct, index):
        n_range = torch.arange(self.n, device=self.device)
        structure_info = torch.zeros((self.n, self.library_size), device=self.device)
        index_val = prev_struct[n_range, index].argmax(dim=-1).unsqueeze(1)

        two_child_struct = torch.zeros((self.n, self.library_size), device=self.device)
        two_child_struct[n_range, :self.two_children_num] = 1.0
        structure_info += (index_val < self.two_children_num).float()* two_child_struct

        one_child_struct = torch.zeros((self.n, self.library_size), device=self.device)
        one_child_struct[n_range, self.two_children_num:self.one_children_num] = 1.0
        structure_info += (self.one_children_num > index_val).float() * (index_val>= self.two_children_num).float() * one_child_struct

        variable_struct = torch.zeros((self.n, self.library_size), device=self.device)
        variable_struct[n_range, self.one_children_num:] = 1.0
        structure_info += (index_val >= self.one_children_num).float() * variable_struct

        return structure_info

    # Adds the node to the next valid location in the expression tree according to breadth first traversal
    def add(self, val, node_num):
        n_range = torch.arange(self.n, device=self.device)
        self.bforder[n_range, node_num] = val
        self.is_const[n_range, node_num] = (val == self.library_size - 1)

        self.valid_nodes[n_range, node_num] = (node_num < self.node_counts)
        bools = (self.one_children_num > val) * (val >= self.two_children_num)
        self.rules[n_range, node_num] *= ~((bools.float().unsqueeze(1) @ (~self.ONF_rule).float().unsqueeze(0)).bool())

        node_numbers = self.node_counts.clone()

        self.has_sibling[n_range, node_numbers] = (self.two_children_num > val)
        self.children_locations[n_range, node_num] = node_numbers * (self.one_children_num > val)

        if node_num < self.max_depth - 1:
            self.inputs_backlog[n_range, node_numbers, 0] = val + 1
            self.inputs_backlog[n_range, node_num + 1, 1] = (val + 1) * self.has_sibling[:, node_num] + self.inputs_backlog[n_range, node_num + 1, 1] * (~self.has_sibling[:, node_num])

            self.rules[n_range, node_numbers] = self.rules[n_range, node_num]
            self.positions[n_range, node_numbers, 0] = self.positions[n_range, node_num, 0] + 1
            self.positions[n_range, node_numbers, 1] = self.positions[n_range, node_num, 1] - 1 / torch.pow(2, self.positions[n_range, node_num, 0] + 1)

        if node_num < self.max_depth - 2:
            node_numbers += 1
            self.inputs_backlog[n_range, node_numbers, 0] = (val + 1)

            self.rules[n_range, node_numbers] = self.rules[n_range, node_num]
            self.positions[n_range, node_numbers, 0] = self.positions[n_range, node_num, 0] + 1
            self.positions[n_range, node_numbers, 1] = self.positions[n_range, node_num, 1] + 1 / torch.pow(2, self.positions[n_range, node_num, 0] + 1)

        self.node_counts += (self.one_children_num > val) * (node_num < self.node_counts)
        self.node_counts += (self.two_children_num > val) * (node_num < self.node_counts)

        # self.node_counts = (self.node_counts > 31) * 32 + (self.node_counts <= 31) * self.node_counts
        # Need to add the sibling information and I need to read the code to see if rules is working correctly\

    def update_node(self, position, val):
        n_range = torch.arange(self.n, device=self.device)
        self.bforder[n_range, position] = val
        self.is_const[n_range, position] = (val == self.library_size - 1)

        # Update sibling information
        self.inputs_backlog[n_range, position + 1, 1] = (val + 1) * self.has_sibling[n_range, position] + self.inputs_backlog[n_range, position + 1, 1] * (~self.has_sibling[n_range, position])

        # Update Children information
        has_children = (self.children_locations[n_range, position] != 0)
        children = self.children_locations[n_range, position]
        self.inputs_backlog[n_range, children, 0] = (val + 1) * has_children

        # Update Second Children information
        has_children = (self.children_locations[n_range, position] != 0) * self.has_sibling[n_range, children]
        children = self.children_locations[n_range, position] + 1
        self.inputs_backlog[n_range, children, 0] = (val + 1) * has_children

    # Returns the preorder_traversal
    def get_labels(self):
        lib = F.one_hot(self.bforder.long(), num_classes=self.library_size) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)
        return lib

    # Returns the parent sibling inputs that were used to generate the set
    def get_inputs(self):
        parents = F.one_hot(self.inputs_backlog[:, :self.max_depth, 0].long(), num_classes=self.library_size + 1) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)
        siblings = F.one_hot(self.inputs_backlog[:, :self.max_depth, 1].long(), num_classes=self.library_size + 1) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)
        return torch.cat([parents, siblings], dim=2).bool()

    def get_positions(self):
        return self.positions[:, :self.max_depth, :] * self.valid_nodes[:, :self.max_depth].unsqueeze(2)

    # Fetches the parent and sibling values for the input node_num
    def fetch_ps(self, node_num):
        parents = F.one_hot(self.inputs_backlog[:, node_num, 0].long(), num_classes=self.library_size + 1)
        siblings = F.one_hot(self.inputs_backlog[:, node_num, 1].long(), num_classes=self.library_size + 1)
        return torch.cat([parents, siblings], dim=2).bool()

    # Get the number of nodes in each expression tree
    def get_node_counts(self):
        return self.node_counts

    # Solves for the values for all of the constants in the expression tree
    def opt(self, x_full, y_full, reward_function, opt_lm=True, bic_scaler=1.0):
        x_full = x_full.astype(np.float64)
        y_full = y_full.astype(np.float64)
        data_set_size = len(y_full)
        std = np.std(y_full)
        equations = self.equation_string()

        if reward_function.value == RewardFunctions.NMSE.value:
            reward = NMSE_reward_func
        elif reward_function.value == RewardFunctions.BIC.value:
            reward = BIC_np_calc_loss
        elif reward_function.value == RewardFunctions.RegNMSE.value:
            reward = NMSE_reg_reward_func
        elif reward_function.value == RewardFunctions.SPLReward.value:
            reward = SPL_reg_reward_func
        else:
            reward = calc_r_squared

        if self.max_cpu_count > 1:
            # Prepare tasks for multiprocessing
            tasks = []
            for i in range(self.n):
                tasks.append((
                    i,
                    equations[i],
                    self.constants[i],
                    self.incremental_constant[i],
                    (self.bforder[i] == 2).sum().item() if hasattr(self.bforder[i], 'sum') else (self.bforder[i] == 2).float().sum().item(),
                    self.node_counts[i].item()
                ))

            # Process equations in parallel
            with Pool(
                    initializer=init_pool,
                    initargs=(x_full, y_full),
                    processes=min(os.cpu_count(), self.max_cpu_count)  # Use available cores
            ) as pool:
                results = pool.starmap(
                    process_equation,
                    [(task, self.max_dataset_size, reward, opt_lm, std, bic_scaler, reward_function.value) for task in tasks]
                )

            # Update class state with results
            for result in results:
                i, new_const, reward_val, noise_val, error_message = result
                self.constants[i] = new_const
                self.rewards[i] = reward_val
                self.noise[i] = noise_val
                # if error_message is not None:
                #     print(error_message)
        else:
            for i in range(self.n):
                if data_set_size > self.max_dataset_size:
                    perm = np.random.permutation(data_set_size)[:self.max_dataset_size]
                    x = x_full.T[perm].T
                    y = y_full[perm]
                else:
                    x = x_full
                    y = y_full
                try:
                    prod_count = (self.bforder[i, :] == 2).float().sum()
                    # Checking to see if there are no constants
                    if self.incremental_constant[i] == 0:
                        c = self.constants[i]
                        x = x_full
                        y = y_full
                        pred_y = eval(equations[i])
                        v = np.mean((pred_y - y) ** 2)
                        self.noise[i] = v
                        self.rewards[i] = reward(pred_y, y, std, v, self.incremental_constant[i] if reward_function.value != RewardFunctions.SPLReward.value else prod_count, self.node_counts[i].item(), bic_scaler)
                        if np.isnan(self.rewards[i]) or np.iscomplex(self.rewards[i]):
                            self.rewards[i] = np.nan
                        continue

                    # There is a negative sign on the return change the reward for being maximized to being minimized
                    def ls_func(c):
                        nonlocal x
                        return y - eval(equations[i])

                    def min_func(c):
                        nonlocal x
                        return np.sum((y - eval(equations[i]))**2)

                    if opt_lm:
                        info = optimize.least_squares(ls_func, self.constants[i], method='lm')
                    else:
                        info = optimize.minimize(min_func, self.constants[i], method='L-BFGS-B')
                    self.constants[i] = info.x
                    c = self.constants[i]
                    x = x_full
                    y = y_full
                    pred_y = eval(equations[i])
                    v = np.mean((pred_y - y) ** 2)
                    self.noise[i] = v
                    self.rewards[i] = reward(pred_y, y, std, v, self.incremental_constant[i] if reward_function != RewardFunctions.SPLReward.value else prod_count, self.node_counts[i].item(), bic_scaler)

                except(ZeroDivisionError, ValueError, TypeError, OverflowError):
                    self.rewards[i] = np.nan

        if reward_function.value == RewardFunctions.BIC.value:
            self.rewards = -np.nan_to_num(self.rewards, nan=np.inf)
        else:
            self.rewards = np.nan_to_num(self.rewards, nan=0)

        return self.rewards

    def calc_r2s(self, x, y):
        r_2s = []
        mu = np.mean(y)
        normalizer = np.sum((y - mu) ** 2)
        equations = self.equation_string()
        for i, equation in enumerate(equations):
            if self.rewards[i] == -torch.inf:
                r_2s.append(-np.inf)
                continue
            try:
                c = self.constants[i]
                device = self.device
                pred_y = eval(equation)
                r2 = calc_r_squared(pred_y, y, normalizer)
                if np.isnan(r2):
                    r_2s.append(-np.inf)
                else:
                    r_2s.append(r2)
            except:
                r_2s.append(-np.inf)
        return r_2s

    def fetch_rules(self, node_num):
        # Need to add constant rule here
        n_range = torch.arange(self.n, device=self.device)
        bools = (self.node_counts == self.max_depth - 1)
        self.rules[n_range, node_num] *= ~((bools.float().unsqueeze(1) @ (~self.one_func_or_vars_rule).float().unsqueeze(0)).bool())
        bools = (self.node_counts == self.max_depth)
        self.rules[n_range, node_num] *= ~((bools.float().unsqueeze(1) @ (~self.vars_rule).float().unsqueeze(0)).bool())
        if self.opt_const:
            exceeded_max_const = ((self.bforder == self.library_size-1).sum(dim=1) < self.max_num_const).float()
            self.rules[n_range, node_num, -1] = (exceeded_max_const * self.rules[n_range, node_num, -1]).bool()
        return self.rules[n_range, node_num]

    def reduce(self, indices):
        self.n = len(indices)
        self.rewards = [self.rewards[i] for i in indices]
        self.constants = [self.constants[i] for i in indices]
        self.noise = [self.noise[i] for i in indices]
        self.incremental_constant = [self.incremental_constant[i] for i in indices]
        if len(self.equations) != 0:
            self.equations = [self.equations[i] for i in indices]

        indices_tensor = torch.tensor(indices.copy(), device=self.device, dtype=torch.int32)
        self.positions_history = torch.index_select(self.positions_history, dim=0, index=indices_tensor)
        self.bforder = torch.index_select(self.bforder, dim=0, index=indices_tensor)
        self.children_locations = torch.index_select(self.children_locations, dim=0, index=indices_tensor)
        self.has_sibling = torch.index_select(self.has_sibling, dim=0, index=indices_tensor)
        self.inputs_backlog = torch.index_select(self.inputs_backlog, dim=0, index=indices_tensor)
        self.positions = torch.index_select(self.positions, dim=0, index=indices_tensor)
        self.node_counts = torch.index_select(self.node_counts, dim=0, index=indices_tensor)
        self.valid_nodes = torch.index_select(self.valid_nodes, dim=0, index=indices_tensor)
        self.is_const = torch.index_select(self.is_const, dim=0, index=indices_tensor)
        self.is_const_parent = torch.index_select(self.is_const_parent, dim=0, index=indices_tensor)
        self.is_const_sibling = torch.index_select(self.is_const_sibling, dim=0, index=indices_tensor)
        self.rules = torch.index_select(self.rules, dim=0, index=indices_tensor)

    def join(self, trees):
        self.n += trees.n
        self.constants += trees.constants
        self.noise += trees.noise
        self.incremental_constant += trees.incremental_constant
        self.equations += trees.equations

        self.rewards = np.concatenate((self.rewards, trees.rewards), axis=0)

        self.positions_history = torch.cat((self.positions_history, trees.positions_history), dim=0)
        self.bforder = torch.cat((self.bforder, trees.bforder), dim=0)
        self.children_locations = torch.cat((self.children_locations, trees.children_locations), dim=0)
        self.has_sibling = torch.cat((self.has_sibling, trees.has_sibling), dim=0)
        self.inputs_backlog = torch.cat((self.inputs_backlog, trees.inputs_backlog), dim=0)
        self.positions = torch.cat((self.positions, trees.positions), dim=0)
        self.node_counts = torch.cat((self.node_counts, trees.node_counts), dim=0)
        self.valid_nodes = torch.cat((self.valid_nodes, trees.valid_nodes), dim=0)
        self.is_const = torch.cat((self.is_const, trees.is_const), dim=0)
        self.is_const_parent = torch.cat((self.is_const_parent, trees.is_const_parent), dim=0)
        self.is_const_sibling = torch.cat((self.is_const_sibling, trees.is_const_sibling), dim=0)
        self.rules = torch.cat((self.rules, trees.rules), dim=0)

    def duplicate(self, t):
        indices = [int(i/t) for i in range(t * self.n)]
        self.reduce(indices)

    def unique(self):
        if self.equations is None:
            return
        unique = []
        sample_equs = {}
        for index in range(self.n):
            equ = self.equations[index]
            if equ not in sample_equs:
                unique.append(index)
                sample_equs[equ] = True
        self.reduce(unique)


def categorical_sample(x):
    x = (x / torch.sum(x, dim=1, keepdim=True))
    if x.isnan().any():
        raise ValueError("Nans in prob")
    return Categorical(x).sample()
