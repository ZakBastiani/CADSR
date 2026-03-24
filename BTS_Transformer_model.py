import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import numpy as np
from expression_tree import ExpressionTree
from enums import *
from position_encodings import *
from scipy.optimize import linear_sum_assignment


class BTSTransformerModel(nn.Module):
    def __init__(self, two_children_funcs, one_children_funcs, variables, **kwargs):
        super(BTSTransformerModel, self).__init__()
        default_model_parameters = {
            "max_depth": 32,
            "num_heads": 1,
            "dim_feedforward": 2048,
            "encode_layers": 0,
            "decoder_layers": 1,
            "oversampling": 3,
            "opt_const": True,
            "use_dct": True,
            "embedding_dim": 10,
            "dct_dim": 8,
            "device": torch.cuda,
            "max_num_const": 10,
            "diff_steps": 1,
            "max_layers": 1
        }
        for key, value in default_model_parameters.items():
            if key not in kwargs.keys():
                kwargs[key] = value

        self.device = kwargs["device"]
        self.opt_const = kwargs["opt_const"]
        self.max_depth = kwargs["max_depth"]
        self.oversampling_scalar = kwargs["oversampling"]
        self.two_children_funcs = two_children_funcs
        self.two_children_num = len(two_children_funcs)
        self.one_children_funcs = one_children_funcs
        self.one_children_num = len(one_children_funcs) + len(two_children_funcs)
        self.variables = variables
        self.max_num_const = kwargs["max_num_const"]
        self.max_layers = kwargs["max_layers"]

        self.library_size = len(self.two_children_funcs) + len(self.one_children_funcs) + len(self.variables)
        self.input_size = 2 * (self.library_size + 1)
        self.label_size = self.library_size
        self.embedding_dim = kwargs["embedding_dim"]
        self.dct_dim = kwargs["dct_dim"]

        if kwargs["use_dct"]:
            self.dct_matrix = create_dct(self.embedding_dim, self.dct_dim).to(self.device)
        else:
            self.dct_matrix = None
            self.dct_dim = self.embedding_dim

        # self.position = OneDimensionalPositionalEncoding(d_model=self.input_size, max_len=max_depth)

        self.ps_embedding = nn.Linear(in_features=self.input_size, out_features=self.embedding_dim)
        self.target_embedding = nn.Linear(in_features=self.library_size + 1, out_features=self.embedding_dim)

        self.mask = self.generate_square_subsequent_mask(self.max_depth)

        if kwargs["encoder_layers"] != 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.dct_dim, dim_feedforward=kwargs["dim_feedforward"], nhead=kwargs["num_heads"], dropout=0,
                                                       batch_first=True, norm_first=True)

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=kwargs["encoder_layers"])
        else:
            self.encoder = None

        if kwargs["decoder_layers"] != 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.dct_dim, dim_feedforward=kwargs["dim_feedforward"], nhead=kwargs["num_heads"], dropout=0,
                                                       batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=kwargs["decoder_layers"])
        else:
            self.decoder = None

        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.library_size)

        self.softmax = nn.Softmax(dim=2)

        self.scr_mask = self.generate_square_subsequent_mask(self.max_depth)

        self.tgt_mask = self.generate_square_subsequent_mask(self.max_depth)

        # Weight initialization
        for name, param in self.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.xavier_uniform_(param)

    def generate_square_subsequent_mask(self, size):  # Generate mask covering the top right triangle of a matrix
        mask = torch.triu(torch.full((size, size), float('-inf'), device=self.device), diagonal=1)
        return mask


class AutoregressiveModel(BTSTransformerModel):
    def __init__(self, two_children_funcs, one_children_funcs, variables, **kwargs):
        super(AutoregressiveModel, self).__init__(two_children_funcs, one_children_funcs, variables, **kwargs)

        if "dpo_split" not in kwargs.keys():
            kwargs["dpo_split"] = 3

        if "pe" not in kwargs.keys():
            kwargs["pe"] = PositionalEncodings.TwoDPE

        if PositionalEncodings.TwoDPE == kwargs["pe"]:
            self.position = TwoDimensionalPositionalEncoding(d_model=self.embedding_dim, max_len=kwargs["max_depth"], device=self.device)
        elif PositionalEncodings.OneDPE == kwargs["pe"]:
            self.position = OneDimensionalPositionalEncoding(d_model=self.embedding_dim, max_len=kwargs["max_depth"], device=self.device)
        else:
            self.position = NoPositionalEncoding()

        self.dpo_split = kwargs["dpo_split"]

    def forward(self, targets, ps_information, p, temp=1):

        ps_information = self.ps_embedding(ps_information)
        ps_information = self.position(ps_information, p)

        targets = right_shift(targets)
        targets = self.target_embedding(targets)
        targets = self.position(targets, p)

        if self.dct_matrix is not None:
            ps_information = ps_information @ self.dct_matrix.T
            targets = targets @ self.dct_matrix.T

        if self.encoder is not None:
            encoder_info = self.encoder(ps_information, mask=self.scr_mask)
        else:
            encoder_info = targets

        x = self.decoder(tgt=targets, memory=encoder_info, tgt_mask=self.tgt_mask, memory_mask=self.scr_mask)

        if self.dct_matrix is not None:
            x = x @ self.dct_matrix

        x = self.linear(x)

        labels = self.softmax(x / temp)
        return labels

    def sample(self, n, device):
        sample_equs = {}
        with torch.no_grad():
            dictionary = {"Fetch PS Time": 0, "Prediction Time": 0, "Apply Rules Time": 0, "Rand_Cat Time": 0,
                          "Add Node Time": 0, "Build Time": 0, "Equation Build Time": 0, "Comparison Time": 0}

            a = time.time()
            trees = ExpressionTree(n=int(self.oversampling_scalar * n), two_children_funcs=self.two_children_funcs,
                                   one_children_funcs=self.one_children_funcs, variables=self.variables,
                                   max_depth=self.max_depth, max_num_const=self.max_num_const, opt_const=self.opt_const, device=device)
            torch.cuda.synchronize()
            dictionary["Build Time"] += time.time() - a
            for j in range(self.max_depth):
                torch.cuda.synchronize()
                a = time.time()
                ps_info = trees.get_inputs().float().to(self.device)
                targets = trees.get_labels().float().to(self.device)
                positions = trees.get_positions().float().to(self.device)

                torch.cuda.synchronize()
                dictionary["Fetch PS Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                x = self.forward(targets, ps_info, positions, temp=1)[:, j, :] + 1E-5
                x = x.to(device)
                torch.cuda.synchronize()
                dictionary["Prediction Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                rules = trees.fetch_rules(j)

                x = x * rules
                torch.cuda.synchronize()
                dictionary["Apply Rules Time"] += time.time() - a
                torch.cuda.synchronize()
                a = time.time()
                predicted_vals = categorical_sample(x)

                torch.cuda.synchronize()
                dictionary["Rand_Cat Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                trees.add(predicted_vals.int(), j)
                torch.cuda.synchronize()
                dictionary["Add Node Time"] += time.time() - a

            torch.cuda.synchronize()
            a = time.time()
            equations = trees.equation_string()
            torch.cuda.synchronize()
            dictionary["Equation Build Time"] += time.time() - a
            torch.cuda.synchronize()
            a = time.time()
            unique = []
            i = 0
            for index in range(int(self.oversampling_scalar * n)):
                equ = equations[index]
                if equ not in sample_equs:
                    unique.append(index)
                    sample_equs[equ] = -torch.inf
                    i += 1
                    if i == n:
                        break
                elif n - i >= self.oversampling_scalar * n - index:
                    unique.append(index)
                    i += 1

            trees.reduce(unique)
            torch.cuda.synchronize()
            dictionary["Comparison Time"] += time.time() - a

            # print(dictionary)
            return trees, dictionary

    def dpo_sample(self, n, device):

        with torch.no_grad():
            dictionary = {"Fetch PS Time": 0, "Prediction Time": 0, "Apply Rules Time": 0, "Rand_Cat Time": 0,
                          "Add Node Time": 0, "Build Time": 0, "Equation Build Time": 0, "Comparison Time": 0}

            a = time.time()
            trees = ExpressionTree(n=int(n), two_children_funcs=self.two_children_funcs,
                                   one_children_funcs=self.one_children_funcs, variables=self.variables,
                                   max_depth=self.max_depth, max_num_const=self.max_num_const, opt_const=self.opt_const, device=device)
            torch.cuda.synchronize()
            dictionary["Build Time"] += time.time() - a
            for j in range(self.dpo_split):
                torch.cuda.synchronize()
                a = time.time()
                ps_info = trees.get_inputs().float().to(self.device)
                targets = trees.get_labels().float().to(self.device)
                positions = trees.get_positions().float().to(self.device)

                torch.cuda.synchronize()
                dictionary["Fetch PS Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                x = self.forward(targets, ps_info, positions, temp=1)[:, j, :] + 1E-5
                x = x.to(device)
                torch.cuda.synchronize()
                dictionary["Prediction Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                rules = trees.fetch_rules(j)

                x = x * rules * (~trees.vars_rule)
                torch.cuda.synchronize()
                dictionary["Apply Rules Time"] += time.time() - a
                torch.cuda.synchronize()
                a = time.time()
                predicted_vals = categorical_sample(x)

                torch.cuda.synchronize()
                dictionary["Rand_Cat Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                trees.add(predicted_vals.int(), j)
                torch.cuda.synchronize()
                dictionary["Add Node Time"] += time.time() - a

            trees.duplicate(2)

            for j in range(self.dpo_split, self.max_depth):
                torch.cuda.synchronize()
                a = time.time()
                ps_info = trees.get_inputs().float().to(self.device)
                targets = trees.get_labels().float().to(self.device)
                positions = trees.get_positions().float().to(self.device)

                torch.cuda.synchronize()
                dictionary["Fetch PS Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                x = self.forward(targets, ps_info, positions, temp=1)[:, j, :] + 1E-5
                x = x.to(device)
                torch.cuda.synchronize()
                dictionary["Prediction Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                rules = trees.fetch_rules(j)

                x = x * rules
                torch.cuda.synchronize()
                dictionary["Apply Rules Time"] += time.time() - a
                torch.cuda.synchronize()
                a = time.time()
                predicted_vals = categorical_sample(x)

                torch.cuda.synchronize()
                dictionary["Rand_Cat Time"] += time.time() - a

                torch.cuda.synchronize()
                a = time.time()
                trees.add(predicted_vals.int(), j)
                torch.cuda.synchronize()
                dictionary["Add Node Time"] += time.time() - a

            torch.cuda.synchronize()
            a = time.time()
            trees.equation_string()
            torch.cuda.synchronize()
            dictionary["Equation Build Time"] += time.time() - a
            torch.cuda.synchronize()

            return trees, dictionary

def categorical_sample(x):
    x = (x / torch.sum(x, dim=1, keepdim=True))
    return Categorical(x).sample()


def right_shift(targets):
    padded_targets = F.pad(targets, (1, 0, 1, 0, 0, 0), "constant", 0)
    padded_targets[:, 0, 0] = 1
    return padded_targets[:, :-1]


def dct(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    x = src.clone()
    N = x.shape[dim]

    x = x.transpose(dim, -1)
    x_shape = x.shape
    x = x.contiguous().view(-1, N)

    v = torch.empty_like(x, device=x.device)
    v[..., :(N - 1) // 2 + 1] = x[..., ::2]

    if N % 2:  # odd length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., 1::2]
    else:  # even length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., ::2]

    V = torch.fft.fft(v, dim=-1)

    k = torch.arange(N, device=x.device)
    V = 2 * V * torch.exp(-1j * np.pi * k / (2 * N))

    if norm == 'ortho':
        V[..., 0] *= math.sqrt(1 / (4 * N))
        V[..., 1:] *= math.sqrt(1 / (2 * N))

    V = V.real
    V = V.view(*x_shape).transpose(-1, dim)

    return V


def idct(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    X = src.clone()
    N = X.shape[dim]

    X = X.transpose(dim, -1)
    X_shape = X.shape
    X = X.contiguous().view(-1, N)

    if norm == 'ortho':
        X[..., 0] *= 1 / math.sqrt(2)
        X *= N * math.sqrt((2 / N))
    else:
        raise Exception("idct with norm=None is buggy A.F")

    k = torch.arange(N, device=X.device)

    X = X * torch.exp(1j * np.pi * k / (2 * N))
    X = torch.fft.ifft(X, dim=-1).real
    v = torch.empty_like(X, device=X.device)

    v[..., ::2] = X[..., :(N - 1) // 2 + 1]
    v[..., 1::2] = X[..., (N - 1) // 2 + 1:].flip(-1)

    v = v.view(*X_shape).transpose(-1, dim)

    return v


def create_dct(n, m=None):
    I = torch.eye(n)
    Q = dct(I, dim=0)

    if m is not None:
        Q = Q[:m, :]

    return Q
