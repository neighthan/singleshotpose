from typing import Any, Dict

from comet_ml import Experiment


class Logger:
    def log(self, metrics: Dict[str, Any]):
        """
        :param metrics:
        """


class StdIOLogger(Logger):
    def log(self, metrics):
        print(
            "{n_seen}: nGT {n_ground_truth}, recall {n_correct}, proposals {n_proposals}, loss: x {loss_x}, y {loss_y}, conf {loss_conf}, cls {loss_cls}, total {loss}".format(
                **metrics
            )
        )


class CometLogger(Logger):
    def __init__(self, experiment: Experiment):
        self.exp = experiment

    def log(self, metrics):
        self.exp.log_metrics(metrics)
