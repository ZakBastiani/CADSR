from enum import Enum


class RiskSeekingPolicyScaling(Enum):
    Linear = "Linear"
    Uniform = "Uniform"
    RunningMean = "RunningMean"
    Exponential = "Exponential"


class Policies(Enum):
    Basic = "Basic"
    PPO = "PPO"
    DPO = "DPO"
    GRPO = "GRPO"


class OptimizationTypes(Enum):
    LM = "lm"
    BFGS = "BFGS"


class PositionalEncodings(Enum):
    TwoDPE = "TwoDPE"
    OneDPE = "OneDPE"
    NoPE = "NoPE"


class RewardFunctions(Enum):
    NMSE = "NMSE"
    RegNMSE = "RegNMSE"
    BIC = "BIC"
    SPLReward = "SPLReward"
    R2 = "R2"


class SamplingMethod(Enum):
    Autoregressive = "Autoregressive"

