import logging
import os
from typing import Any

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    DirectionReconstructionWithKappa,
    ZenithReconstructionWithKappa,
)
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from torch.optim.adam import Adam

from .data_loader import make_test_dataloader

log = logging.getLogger(__name__)


def build_model(c, dataloader: Any) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
    )

    task = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=c.model_params.target,
        loss_function=VonMisesFisher3DLoss(),
    )
    prediction_columns = [
        c.model_params.target + "_x",
        c.model_params.target + "_y",
        c.model_params.target + "_z",
        c.model_params.target + "_kappa",
    ]
    additional_attributes = ["zenith", "azimuth", "event_id"]

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={
            "lr": c.training_params.lr,
            "eps": c.training_params.eps,
            "weight_decay": c.training_params.weight_decay,
        },
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(dataloader) / 2,
                len(dataloader) * c.training_params.epoch,
            ],
            "factors": [1e-02, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes

    return model


def load_pretrained_model(
    c,
    state_dict_path: str = "dynedge_pretrained_batch_1_to_50/state_dict.pth",
) -> StandardModel:
    test_loader = make_test_dataloader(c)
    model = build_model(c, dataloader=test_loader)
    # model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(os.path.join(c.data.dir.pretrained, state_dict_path))
    model.prediction_columns = [
        c.model_params.target + "_x",
        c.model_params.target + "_y",
        c.model_params.target + "_z",
        c.model_params.target + "_kappa",
    ]
    model.additional_attributes = ["event_id"]  #'zenith', 'azimuth',  not available in test data
    return model
