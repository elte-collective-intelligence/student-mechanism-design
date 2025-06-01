import logging
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import torch
class Logger:
    """Logger for training progress, rewards, and other metrics with Weights & Biases integration."""

    def __init__(
        self, 
        wandb_api_key=None,
        wandb_project=None, 
        wandb_entity=None, 
        wandb_config=None, 
        wandb_run_name=None,
        wandb_resume=False,
        configs={}
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
        self.logger.setLevel(logging.DEBUG)
        self.configs = configs
        self.log_dir = self.configs["log_dir"]
        os.makedirs(self.configs["log_dir"], exist_ok=True)
        self.use_wandb=False
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(self.configs["log_dir"], self.configs["log_file"]))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # self.logger.addHandler(fh)

        # TensorBoard handler
        self.writer = SummaryWriter(log_dir=self.configs["log_dir"])

        # Weight and biasses
        if wandb_project != "" and wandb_api_key != "" and wandb_entity != "":
            wandb_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "config": wandb_config,
                "name": wandb_run_name,
                "dir": self.configs["log_dir"],
                "resume": "allow" if wandb_resume else False
            }
            
            # Remove None values to avoid wandb warnings
            wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
            wandb.login(key=wandb_api_key)
            wandb.init(**wandb_kwargs)
            self.logger.info("Weights & Biases initialized.")
            self.use_wandb = True
            wandb.define_metric("epoch_step")
            wandb.define_metric("episode_step")
            wandb.define_metric("epoch/", step_metric="epoch_step")
            wandb.define_metric("reward_weight/", step_metric="epoch_step")
            wandb.define_metric("episode/", step_metric="episode_step")
            self.log("Using Wandb initialized.","info")
    def log(self, message, level='info'):
        """
        Log a message to console and file.

        Args:
            message (str): The message to log.
            level (str, optional): The log level ('info', 'warning', 'error').
        """
        if level == 'info' or self.configs["verbose"] == True:
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

    def log_scalar(self, tag, value, step=None):
        """
        Log a scalar value to TensorBoard and Weights & Biases.

        Args:
            tag (str): Identifier for the scalar.
            value (float): Value to log.
            step (int): Step number.
        """
        # self.log(str(step) + " | " + tag + ": " + str(value), level='info')
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb and wandb.run:
            wandb.log({tag: value}, step=step)

    def log_weights(self, weights, step=None):
        """
        Log model weights to TensorBoard.

        Args:
            weights (dict): Dictionary of model weights.
            step (int): Step number.
        """
        for name, param in weights.items():
            self.log_scalar("reward_weight/"+name, param, step)

    def log_plt(self, name, plt, step=None):
        if self.use_wandb and wandb.run:
            wandb.log({name: wandb.Image(plt)})

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
        if self.use_wandb and wandb.run:
            wandb.log(metrics, step=step)

    def log_hyperparameters(self, params):
        """
        Log hyperparameters to Weights & Biases.

        Args:
            params (dict): A dictionary of hyperparameter names and their values.
        """
        if self.use_wandb and wandb.run:
            wandb.config.update(params)

    def log_model(self, model, model_name):
        """
        Log model architecture to Weights & Biases.

        Args:
            model (torch.nn.Module): The model to log.
        """
        path =  self.log_dir + "/" + model_name + ".pt"
        torch.save(model.state_dict(), path)
        if self.use_wandb and wandb.run:
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def load_model(self, model_name, model_num=None ):
        """
        Load model from Weights & Biases.

        Args:
            model_name (str): The name of the model.
        """
        if self.use_wandb and wandb.run:
            if(model_num):
                model_name_with_num = model_name + ":" + str(model_num) #this functionality is not included, fyi
            else:
                model_name_with_num = model_name + ":latest"
            #artifact = wandb.use_artifact(model_name_with_num, type='model')
            #model_dir = artifact.download()
            model_dir = self.log_dir
        else:
            model_dir = self.log_dir
        
        model = torch.load(model_dir + "/" + model_name + ".pt")
        return model
    def close(self):
        """
        Close the TensorBoard writer and Weights & Biases run.
        """
        self.writer.close()
        if self.use_wandb and wandb.run:
            wandb.finish()

    def model_exists(self, model_name):
        return os.path.exists(f"{os.path.join(self.log_dir, model_name)}.pt")