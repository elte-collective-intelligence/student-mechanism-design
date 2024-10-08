import logging
from torch.utils.tensorboard import SummaryWriter
import wandb
import os

class Logger:
    """Logger for training progress, rewards, and other metrics with Weights & Biases integration."""

    def __init__(
        self, 
        log_dir='logs', 
        wandb_api_key=None,
        wandb_project=None, 
        wandb_entity=None, 
        wandb_config=None, 
        wandb_run_name=None,
        wandb_resume=False
    ):
        """
        Initializes the Logger with console, file, TensorBoard, and optionally Weights & Biases logging.

        Args:
            log_dir (str): Directory where logs and TensorBoard files will be saved.
            wandb_project (str, optional): Name of the wandb project. If None, wandb won't be initialized.
            wandb_entity (str, optional): The wandb entity (user or team) under which to log.
            wandb_config (dict, optional): Configuration parameters to log to wandb.
            wandb_run_name (str, optional): Custom name for the wandb run.
            wandb_resume (bool, optional): Whether to resume the wandb run if it exists.
        """
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)

        os.makedirs(log_dir, exist_ok=True)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # TensorBoard handler
        self.writer = SummaryWriter(log_dir=log_dir)

        # Weight and biasses
        if wandb_project:
            wandb_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "config": wandb_config,
                "name": wandb_run_name,
                "dir": log_dir,
                "resume": "allow" if wandb_resume else False
            }
            
            # Remove None values to avoid wandb warnings
            wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
            wandb.login(key=wandb_api_key)
            wandb.init(**wandb_kwargs)
            self.logger.info("Weights & Biases initialized.")

    def log(self, message, level='info'):
        """
        Log a message to console and file.

        Args:
            message (str): The message to log.
            level (str, optional): The log level ('info', 'warning', 'error').
        """
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

    def log_scalar(self, tag, value, step):
        """
        Log a scalar value to TensorBoard and Weights & Biases.

        Args:
            tag (str): Identifier for the scalar.
            value (float): Value to log.
            step (int): Step number.
        """
        self.writer.add_scalar(tag, value, step)
        if wandb.run:
            wandb.log({tag: value}, step=step)

    def log_metrics(self, metrics, step=None):
        """
        Log multiple metrics to TensorBoard and Weights & Biases.

        Args:
            metrics (dict): A dictionary of metric names and their corresponding values.
            step (int, optional): Step number.
        """

        # Log to TensorBoard
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

        # Log to Wandb
        if wandb.run:
            wandb.log(metrics, step=step)

    def log_hyperparameters(self, params):
        """
        Log hyperparameters to Weights & Biases.

        Args:
            params (dict): A dictionary of hyperparameter names and their values.
        """
        if wandb.run:
            wandb.config.update(params)

    def close(self):
        """
        Close the TensorBoard writer and Weights & Biases run.
        """
        self.writer.close()
        if wandb.run:
            wandb.finish()
