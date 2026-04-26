from dataclasses import dataclass

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE DO NOT EDIT/MODIFY THIS FILE
# If you want to use other argument values than the defaults, specify them
# in the command line. For instance: 
# To launch the program with 2 runs, just run:
#
#           accelerate launch runproject.py --n_runs=2
#

@dataclass
class Config:
    """
    Dataclass for the script arguments
    """
    # General options
    ollama_url: str = 'http://localhost:11434/v1'
    ollama_url: str = "http://chaos-04.int.europe.naverlabs.com:11434/v1"
    ollama_model: str = "gemma3:4b"
    #
    eval_batch_size: int = 10
    n_runs: int = 5
    # n_train is the number of samples on which to train. n_train=-1 means train all train data
    n_train: int = -1
    # n_eval is the number of samples on which to run the eval. n_eval=-1 means eval on all data samples
    n_eval: int = -1

    #### Training parameters ####
    num_epochs : int = 4
    num_epochs_head: int = 30
    train_batch_size : int = 32
    learning_rate: float = 0.00007096281310557884
    head_learning_rate: float = 0.0009070036730760508
    weight_decay: float = 0.00014901968993233502
    grad_acc: int = 2
    mix_alpha: float = 0.3640517432260951
    mix_prob: float = 0.400956714339144
    tau : float = 0.95


