import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)

import numpy as np

from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from pycls.models.resnet import resnet18
import pycls.core.optimizer as optim

import numpy as np
import yaml
from ConfigSpace import ConfigurationSpace, Float, Categorical, Integer, EqualsCondition
from omegaconf import DictConfig, OmegaConf
from yacs.config import CfgNode

import yaml
from pycls.utils.meters import BufferedFileLogger

from yacs.config import CfgNode

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmacTuner:
    def __init__(self, cfg, train_func, lSet_loader, valSet_loader, cur_episode,
                 intensifier_kwargs={"initial_budget": 5, "max_budget": 30, "eta": 3},
                 scenario_kwargs={"runcount-limit": 200}):
        self.train_model = train_func

        self.experiment_cfg = cfg
        cfg = yaml.safe_load(cfg.dump())
        self.base_config = cfg
        self.g = self.convert_dot_notation_to_nested(cfg)
        self.lSet_loader = lSet_loader
        self.valSet_loader = valSet_loader
        self.cur_episode = cur_episode
        self.intensifier_kwargs = intensifier_kwargs
        self.scenario_kwargs = scenario_kwargs
        self.file_buffer = BufferedFileLogger(
                file_name='smac_optimization.csv',
                file_path=self.experiment_cfg.EXP_DIR,
                buffer_size=1000,
                header=("config", "performance", "al_step"),
                mode='a'
            )

    @property
    def DCOM_configspace(self):
        cs = ConfigurationSpace(seed=0)

        augment = Categorical("DATASET.AUG_METHOD", ['hflip',
                                                     # 'randaug'
                                                     ], default='hflip')
        # r_n = Integer("RANDAUG.N", (0, 5), default=1, log=False)
        # r_m = Integer("RANDAUG.M", (0, 5), default=5, log=False)
        base_lr = Float("OPTIM.BASE_LR", (0.00001, 0.1), default=0.025, log=True)
        lr_policy = Categorical("OPTIM.LR_POLICY", ['cos',
                                                    # 'exp', 'steps',
                                                    'lin', 'none'], default='cos')
        momentum = Float("OPTIM.MOMENTUM", (0.00001, 0.9), default=0.9, log=True)
        wdecay = Float("OPTIM.WEIGHT_DECAY", (0.00001, 0.1), default=0.0003, log=True)
        gamma = Float("OPTIM.GAMMA", (0.00001, 0.1), default=0.1, log=True)

        # hflip is default, but to use randaug, condition necessary according to:
        # https://github.com/automl-private/TypiClust2/blob /a29b2cc8b6676e56197463b74e0994d9e2c753d3
        # /deep-al/pycls/datasets/data.py#L154
        cs.add_hyperparameters([
            augment,
            # r_n, r_m,
            base_lr, lr_policy, momentum, wdecay, gamma
        ])
        # cs.add_condition(EqualsCondition(r_n, augment, 'randaug'))
        # cs.add_condition(EqualsCondition(r_m, augment, 'randaug'))
        return cs

    def parse_config(self, config_path):
        """read yaml file and return DictConfig object"""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return DictConfig(cfg)

    def convert_dot_notation_to_nested(self, dictionary):
        """
        Example:

            # Example dictionary with dot notation
            example_dict = {
                "dataset.aut_method": "method1",
                "dataset.config.name": "example_config",
                "dataset.config.version": 1,
            }

            # Convert to nested dictionary
            nested_dict = convert_dot_notation_to_nested(example_dict)

            # Convert nested dictionary to YAML
            yaml_output = yaml.dump(nested_dict, default_flow_style=False)
            print(yaml_output)
        """
        nested_dict = {}
        for key, value in dictionary.items():
            keys = key.split(".")
            d = nested_dict
            for part in keys[:-1]:
                d = d.setdefault(part, {})
            d[keys[-1]] = value
        return nested_dict

    def tae_runner(self, cfg, seed=0, budget=0):

        logger.info(f"Running SMAC with config: {cfg} and budget: {budget}")

        c = self.convert_dot_notation_to_nested(cfg)
        new_cfg = OmegaConf.merge(DictConfig(self.base_config), DictConfig(self.g), DictConfig(c))

        model = resnet18(num_classes=10, use_dropout=True)

        optimizer = optim.construct_optimizer(new_cfg, model)

        with tempfile.TemporaryDirectory() as episode_dir:
            new_cfg.EPISODE_DIR = episode_dir

            new_cfg = OmegaConf.to_container(new_cfg, resolve=True)
            new_cfg = CfgNode(new_cfg)

            best_val_acc, _, checkpoint_file = self.train_model(
                self.lSet_loader, self.valSet_loader,
                model, optimizer, new_cfg,
                self.cur_episode,
                hpopt=True,  # indicator to avoid messing up logs
                max_epoch=int(budget)
            )
        self.file_buffer.add_scalar(f"{str(cfg)}", f"{1 - best_val_acc}", f"{self.cur_episode}")
        self.file_buffer._flush()
        logger.info(f"SMAC finished with validation accuracy: {1- best_val_acc}")
        return 1 - best_val_acc

    def smac_optimize(self):

        cs = self.DCOM_configspace

        # SMAC scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternative to runtime)
                "cs": cs,  # configuration space
                "deterministic": True,
                **self.scenario_kwargs
            }
        )

        # To optimize, we pass the function to the SMAC-object
        smac = SMAC4MF(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self.tae_runner,
            intensifier_kwargs=self.intensifier_kwargs,
        )

        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        logger.info(f"SMAC final incumbent: {incumbent}")

        # convert incumbent configuration to CfgNode
        c = self.convert_dot_notation_to_nested(incumbent)
        new_cfg = OmegaConf.merge(DictConfig(self.base_config), DictConfig(self.g), DictConfig(c))
        new_cfg = OmegaConf.to_container(new_cfg, resolve=True)
        new_cfg = CfgNode(new_cfg)
        return new_cfg
