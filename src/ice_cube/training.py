from graphnet.training.callbacks import ProgressBar
from pytorch_lightning.callbacks import EarlyStopping


def training(c, train_loader, valid_loader, model):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=c.training_params.es_patience,
        ),
        ProgressBar(),
    ]

    model.fit(
        train_loader,
        valid_loader,
        max_epochs=c.training_params.epoch,
        gpus=[0],
        callbacks=callbacks,
    )

    results = model.predict_as_dataframe(
        valid_loader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
        gpus=[0],
    )
