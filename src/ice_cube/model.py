import logging
import os
from typing import Any, Optional

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    DirectionReconstructionWithKappa,
    ZenithReconstructionWithKappa,
)
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher2DLoss, VonMisesFisher3DLoss
from torch.optim.adam import Adam

from .dynedge import DynEdge

log = logging.getLogger(__name__)


def build_model(c, dataloader: Any, custom_aggregation: bool = False) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "dummy"],
        custom_aggregation=custom_aggregation
    )

    tasks = []
    prediction_columns = []
    additional_attributes = ["zenith", "azimuth", "event_id"]

    if "direction" in c.model_params.tasks:
        direction_task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=c.model_params.target,
            loss_function=VonMisesFisher3DLoss(),
        )
        tasks.append(direction_task)

        prediction_columns += [
            c.model_params.target + "_x",
            c.model_params.target + "_y",
            c.model_params.target + "_z",
            c.model_params.target + "_kappa",
        ]

    if "azimuth" in c.model_params.tasks:
        azimuth_task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels="azimuth",
            loss_function=VonMisesFisher2DLoss(),
        )
        tasks.append(azimuth_task)

        prediction_columns += [
            "azimuth",
            "azimuth_kappa",
        ]
        additional_attributes.remove("azimuth")

    if "zenith" in c.model_params.tasks:
        zenith_task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels="zenith",
            loss_function=VonMisesFisher2DLoss(),
        )
        tasks.append(zenith_task)

        prediction_columns += [
            "zenith",
            "zenith_kappa",
        ]
        additional_attributes.remove("zenith")

    assert tasks != [], "At least one task is required."

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=tasks,
        optimizer_class=Adam,
        optimizer_kwargs={
            "lr": c.training_params.lr,
            "eps": c.training_params.eps,
            # "weight_decay": c.training_params.weight_decay,
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
    dataloader,
    state_dict_path,
    custom_aggregation: bool = False
) -> StandardModel:
    model = build_model(c, dataloader=dataloader, custom_aggregation=custom_aggregation)
    # model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(os.path.join(c.data.dir.pretrained, state_dict_path))

    prediction_columns = []
    if "direction" in c.model_params.tasks:
        prediction_columns += [
            c.model_params.target + "_x",
            c.model_params.target + "_y",
            c.model_params.target + "_z",
            c.model_params.target + "_kappa",
        ]

    if "azimuth" in c.model_params.tasks:
        prediction_columns += [
            "azimuth",
            "azimuth_kappa",
        ]

    if "zenith" in c.model_params.tasks:
        prediction_columns += [
            "zenith",
            "zenith_kappa",
        ]

    model.prediction_columns = prediction_columns
    model.additional_attributes = ["event_id"]  #'zenith', 'azimuth',  not available in test data
    return model
