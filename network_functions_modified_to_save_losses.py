import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd



from provided_code.dose_evaluation_class import DoseEvaluator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
#from keras.models import load_model
#from keras.optimizers.optimizer_v2.adam import Adam



from provided_code.data_loader import DataLoader
from provided_code.network_architectures import DefineDoseFromCT
from provided_code.utils import get_paths, sparse_vector_function


class PredictionModel(DefineDoseFromCT):
    def __init__(self, data_loader: DataLoader, results_patent_path: Path, model_name: str, stage: str) -> None:
        """
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        :param stage: Identify stage of model development (train, validation, test)
        """
        super().__init__(
            data_shapes=data_loader.data_shapes,
            initial_number_of_filters=8,  # Recommend increasing to 64 +
            filter_size=(4, 4, 4), # may change from (4,4,4) to (3,3,3)
            stride_size=(2, 2, 2),
            gen_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), # may change beta_1 from 0.5 to 0.9
        )

        # set attributes for data shape from data loader
        self.generator = None
        self.model_name = model_name
        self.data_loader = data_loader
        self.full_roi_list = data_loader.full_roi_list

        # Define training parameters
        self.current_epoch = 0
        self.last_epoch = 200

        # Make directories for data and models
        model_results_path = results_patent_path / model_name
        self.model_dir = model_results_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir = model_results_path / f"{stage}-predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        # Make template for model path
        self.model_path_template = self.model_dir / "epoch_"

    def train_model(self, epochs: int = 200, save_frequency: int = 5, keep_model_history: int = 2) -> None:
        """
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models are kept on a rolling basis (deletes older than save_frequency * keep_model_history epochs)
        """
        self._set_epoch_start()
        self.last_epoch = epochs
        self.initialize_networks()

        ###
        # ---- NEW: validation data setup ----
        primary_directory = Path().resolve()  # directory where everything is stored
        provided_data_dir = primary_directory / "provided-data"
        validation_data_dir = provided_data_dir / "validation-pats"
        hold_out_data_dir = validation_data_dir
        hold_out_plan_paths = get_paths(hold_out_data_dir)
        self.val_data_loader = DataLoader(hold_out_plan_paths, batch_size=self.data_loader.batch_size)
        self.val_data_loader.set_mode("training_model")
        ###

        ###
        # ---- NEW: training loss file setup ----
        loss_csv_path = self.model_dir.parent / "loss_history.csv"

        if self.current_epoch == 0 and not loss_csv_path.exists():
            with open(loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])
        ###


        if self.current_epoch == epochs:
            print(f"The model has already been trained for {epochs}, so no more training will be done.")
            return
        self.data_loader.set_mode("training_model")
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            print(f"Beginning epoch {self.current_epoch}")
            self.data_loader.shuffle_data()

            ###
            # ---- NEW: training loss accumulator ----
            train_loss_sum = 0.0
            train_batches = 0
            ###

            for idx, batch in enumerate(self.data_loader.get_batches()):
                model_loss = self.generator.train_on_batch([batch.ct, batch.structure_masks], [batch.dose])

                ###
                # ---- NEW: accumulate training loss ----
                train_loss_sum += float(model_loss)
                train_batches += 1
                ###

                print(f"Model loss at epoch {self.current_epoch} batch {idx} is {model_loss:.3f}")

            ###
            # ---- NEW: compute average training loss ----
            epoch_train_loss = train_loss_sum / max(train_batches, 1)
            ###

            ###
            # ---- NEW: validation loss computation ----
            val_loss_sum, n = 0.0, 0
            for batch in self.val_data_loader.get_batches():
                loss = self.generator.test_on_batch([batch.ct, batch.structure_masks], [batch.dose])
                val_loss_sum += float(loss)
                n += 1
            epoch_val_loss = val_loss_sum / max(n, 1)
            ###

            ### ---- NEW: log epoch losses to CSV ----
            with open(loss_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.current_epoch, epoch_train_loss, epoch_val_loss]
                )
            ###

            self.manage_model_storage(save_frequency, keep_model_history)


    def _set_epoch_start(self) -> None:
        all_model_paths = get_paths(self.model_dir, extension="h5")
        for model_path in all_model_paths:
            *_, epoch_number = model_path.stem.split("epoch_")
            if epoch_number.isdigit():
                self.current_epoch = max(self.current_epoch, int(epoch_number))

    def initialize_networks(self) -> None:
        if self.current_epoch >= 1:
            self.generator = load_model(self._get_generator_path(self.current_epoch))
        else:
            self.generator = self.define_generator()

    def manage_model_storage(self, save_frequency: int = 1, keep_model_history: Optional[int] = None) -> None:
        """
        Manage the model storage while models are trained. Note that old models are deleted based on how many models the users has asked to keep.
        We overwrite old files (rather than deleting them) to ensure the Collab users don't fill up their Google Drive trash.
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (older models will be deleted to conserve storage)
        """
        effective_epoch_number = self.current_epoch + 1  # Epoch number + 1 because we're at the start of the next epoch
        if 0 < np.mod(effective_epoch_number, save_frequency) and effective_epoch_number != self.last_epoch:
            Warning(f"Model at the end of epoch {self.current_epoch} was not saved because it is skipped when save frequency {save_frequency}.")
            return

        # The code below is clunky and was only included to bypass the Google Drive trash, which fills quickly with normal save/delete functions
        epoch_to_overwrite = effective_epoch_number - keep_model_history * (save_frequency or float("inf"))
        if epoch_to_overwrite >= 0:
            initial_model_path = self._get_generator_path(epoch_to_overwrite)
            self.generator.save(initial_model_path)
            os.rename(initial_model_path, self._get_generator_path(effective_epoch_number))  # Helps bypass Google Drive trash
        else:  # Save via more conventional method because there is no model to overwrite
            self.generator.save(self._get_generator_path(effective_epoch_number))

    def _get_generator_path(self, epoch: Optional[int] = None) -> Path:
        epoch = epoch or self.current_epoch
        return self.model_dir / f"epoch_{epoch}.h5"

    def predict_dose(self, epoch: int = 1) -> None:
        """Predicts the dose for the given epoch number"""
        self.generator = load_model(self._get_generator_path(epoch))
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print("Predicting dose with generator.")
        for batch in self.data_loader.get_batches():
            dose_pred = self.generator.predict([batch.ct, batch.structure_masks])
            dose_pred = dose_pred * batch.possible_dose_mask
            dose_pred = np.squeeze(dose_pred)
            dose_to_save = sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(data=dose_to_save["data"].squeeze(), index=dose_to_save["indices"].squeeze(), columns=["data"])
            (patient_id,) = batch.patient_list
            dose_df.to_csv("{}/{}.csv".format(self.prediction_dir, patient_id))
