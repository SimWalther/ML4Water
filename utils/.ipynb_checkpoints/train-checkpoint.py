from tensorflow.keras.callbacks import EarlyStopping
from annigma import Annigma
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from feature_groups import FeatureGroups
from callback import (
    VisualizeVegetationWeightsCallback,
    VisualizeGroupCallback,
    VisualizeVegetationWeightsAtAltitudeCallback,
)
from datetime import datetime
import os
import configparser
import keras_tuner
import geopandas as gpd
from spatialkfold.blocks import spatial_blocks
import shap
from tensorflow.keras import Model, layers, Sequential


class Training:
    def __init__(
        self,
        df,
        upstream_slavi_df,
        altitude_diff,
        model,
        config_path,
        save=False,
        output_path="/home/simon.walther/shared-projects/ML4WATER/Uniformise/output",
        station_col="station",
        target_col="water_max_temp",
        feature_groups_config_path="feature_groups.json",
    ):
        self.df = gpd.GeoDataFrame(
            df.reset_index(),
            geometry=gpd.points_from_xy(df.lv95_Longitude, df.lv95_Latitude),
            crs="LV95",
        )

        self.altitude_diff = altitude_diff
        self.upstream_slavi_buffers, self.upstream_slavi_distances = (
            self.extract_buffers_distances_from_upstream_df(upstream_slavi_df)
        )
        self.upstream_slavi_images_dict = self.upstream_features_to_images(
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            upstream_slavi_df,
        )
        self.model = model
        self.feature_groups = FeatureGroups(feature_groups_config_path)
        self.save = save
        self.output_path = output_path
        self.feature_preprocesses = Pipeline([("scaler", StandardScaler())])
        self.context_preprocesses = Pipeline([("scaler", StandardScaler())])
        self.target_col = target_col
        self.station_col = station_col
        self.stations = self.df[self.station_col].unique()
        self.stations_river = [
            self.df.query("station == @station")["river_name"].iloc[0]
            for station in self.stations
        ]
        self.upstream_context_cols = ["altitude_diff"]

        config = configparser.ConfigParser()
        config.read(config_path)

        self.experiment_name = config["General"]["experiment_name"]
        self.n_epochs = int(config["Model"]["n_epochs"])
        self.batch_size = int(config["Model"]["batch_size"])
        self.dataset_ratio_shap = float(config["Model"]["dataset_ratio_shap"])

        self.plot_upstream_feature_images(
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            upstream_slavi_df,
            self.upstream_slavi_images_dict,
        )

    def create_output_folders(self):
        if not os.path.exists(self.experiment_folder):
            os.makedirs(f"{self.experiment_folder}/saved_models")
            os.makedirs(f"{self.experiment_folder}/upstream_attention")
            os.makedirs(f"{self.experiment_folder}/groups_attention")

    def _date_as_str(self):
        return datetime.now().strftime("%Y-%m-%d_%H:%M")

    def plot_upstream_feature_images(
        self,
        buffers,
        distances,
        upstream_df,
        images_dict,
        suptitle="Station Mean Upstream Riparian Tree Coverage",
        cbar_label="Riparian Tree SLAVI-VHM",
        cmap="Greens",
    ):
        year = list(images_dict.keys())[0][1]  # Select first year
        fig, ax = plt.subplots(1, 1, figsize=(7, 2))

        # fig.suptitle(f"{suptitle} ({year})")
        for idx, (station_idx, station_row) in enumerate(
            upstream_df.query(
                "station == 'BOI005' and date.dt.year == @year"
            ).iterrows()
        ):
            ax.set_title(station_row["station"])
            ax.imshow(
                images_dict[(station_row["station"], year)][:, :, 0], cmap="Greens"
            )
            ax.set_xlabel("Upstream distance")
            ax.set_ylabel("Buffer width")
            ax.set_xticks(np.arange(0, len(distances)), distances, rotation=90)
            ax.set_yticks(np.arange(0, len(buffers)), buffers)

        # Add a single colorbar for the entire figure
        cbar = fig.colorbar(
            ax.imshow(
                images_dict[
                    (list(images_dict.keys())[0])[0],
                    list(images_dict.keys())[0][1],
                ][:, :, 0],
                cmap=cmap,
            ),
            ax=ax,
        )
        cbar.set_label(cbar_label)
        plt.show()
        plt.close()

    def train_model_on_all_data(self):
        if self.save:
            self.experiment_folder = (
                f"{self.output_path}/{self.experiment_name + self._date_as_str()}"
            )
            self.create_output_folders()

        train_df = self.df
        train_df = self._drop_nan_rows(train_df)

        # We don't have test so we put train
        data = self.prepare_data(train_df, train_df)

        train_features = data["train_features"]
        test_features = data["test_features"]
        train_target = data["train_target"]
        test_target = data["test_target"]
        train_slavi_images = data["train_slavi_images"]
        test_slavi_images = data["test_slavi_images"]
        train_context = data["train_context"]
        test_context = data["test_context"]

        train_inputs, test_inputs = self.create_model_inputs(
            train_features,
            test_features,
            train_slavi_images,
            test_slavi_images,
            train_context,
            test_context,
        )

        n_samples = 10  # FIXME: move to config
        min_alt = np.quantile(list(self.altitude_diff.values()), 0.25)
        max_alt = np.quantile(list(self.altitude_diff.values()), 0.75)

        tree_context_vis = np.round(
            np.linspace(min_alt, max_alt, n_samples).reshape(-1, 1), 1
        )
        tree_context_vis_preprocessed = self.context_preprocesses.transform(
            tree_context_vis
        )

        callbacks = [
            VisualizeVegetationWeightsCallback(
                train_tree=train_inputs["slavi"],
                context=tree_context_vis,
                context_preprocessed=tree_context_vis_preprocessed,
                distances=self.upstream_slavi_distances,
                buffers=self.upstream_slavi_buffers,
                n_samples=n_samples,
                experiment_folder=self.experiment_folder,
                fold_number=0,
                epoch_to_plot=self.n_epochs - 1,
            ),
            VisualizeVegetationWeightsAtAltitudeCallback(
                train_tree=train_inputs["slavi"],
                context=tree_context_vis,
                context_preprocessed=tree_context_vis_preprocessed,
                distances=self.upstream_slavi_distances,
                buffers=self.upstream_slavi_buffers,
                n_samples=n_samples,
                experiment_folder=self.experiment_folder,
                fold_number=0,
                epoch_to_plot=self.n_epochs - 1,
            ),
            VisualizeGroupCallback(
                self.feature_groups,
                experiment_folder=self.experiment_folder,
                fold_number=0,
                epoch_to_plot=self.n_epochs - 1,
            ),
        ]

        # Build the model
        ann_model = self.model.create(
            self.feature_groups,
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            self.upstream_context_cols,
        )

        history = ann_model.fit(
            train_inputs,
            train_target,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        return (
            ann_model,
            train_inputs,
            history,
        )

    def report_explainable_metrics(self, ann_model, train_inputs, config_path=None):
        if config_path:
            config = configparser.ConfigParser()
            config.read(config_path)
            self.dataset_ratio_shap = float(config["Model"]["dataset_ratio_shap"])

        # Cut model in half (before / after groups)
        for idx, layer in enumerate(ann_model.layers):
            if layer.name == "concatenated_inputs":
                break

        first_half = ann_model.layers[: (idx + 1)]
        first_half_inputs = [
            layer.output
            for layer in first_half
            if layer.__class__.__name__ == "InputLayer"
        ]
        first_half_output = first_half[-1].output
        first_half_model = Model(inputs=first_half_inputs, outputs=first_half_output)

        print("first half model output shape: ", first_half_model.output_shape[-1])
        second_half_input = layers.Input((first_half_model.output_shape[-1],))
        second_half_model = Sequential(
            [second_half_input] + ann_model.layers[(idx + 1) :]
        )

        print("first model: ", first_half_model.summary())
        print("second model: ", second_half_model.summary())

        # Predict all features summarization with the first half model
        predicted_features = first_half_model.predict(train_inputs)
        predicted_features = np.array(predicted_features)

        print("predicted_features shape: ", predicted_features.shape)

        # Background data
        background = predicted_features[
            np.random.choice(predicted_features.shape[0], 1000, replace=False)
        ]
        background = np.stack(background)

        print("background shape: ", background)

        feature_names = np.concatenate(
            [
                (
                    layer.name
                    if layer.name != "input_other"
                    else self.feature_groups.groups["input_other"]
                )
                for layer in first_half
                if layer.__class__.__name__ == "InputLayer" and layer.name != "slope"
            ],
            axis=None,
        )

        print(feature_names)

        explainer = shap.Explainer(
            second_half_model, background, feature_names=feature_names
        )

        # Get SHAP values as an Explanation object directly
        to_explain = predicted_features[
            np.random.choice(
                predicted_features.shape[0],
                int(self.dataset_ratio_shap * len(predicted_features)),
                replace=False,
            )
        ]

        to_explain = np.stack(to_explain)
        shap_values = explainer(to_explain)

        return shap_values, explainer

    def stations_cross_validation(self):
        if self.save:
            self.experiment_folder = (
                f"{self.output_path}/{self.experiment_name + self._date_as_str()}"
            )
            self.create_output_folders()

        # Create folds
        n_folds = 5
        stations_random_blocks = spatial_blocks(
            gdf=self.df,
            width=500,
            height=500,
            method="random",
            nfolds=n_folds,
            random_state=1234,
        )

        self.df = gpd.overlay(self.df, stations_random_blocks)
        fold_models = []
        histories = []

        for fold_number, fold in enumerate(self.df["folds"].unique()):
            print(f"Fold ({fold_number + 1}/{n_folds}): ")

            train_df = self.df.query("folds != @fold")
            test_df = self.df.query("folds == @fold")

            train_df = self._drop_nan_rows(train_df)
            test_df = self._drop_nan_rows(test_df)

            fold_model, history = self.train_fold(fold_number, train_df, test_df)
            fold_models.append(fold_model)
            histories.append(history)

        return fold_models, histories

    def leave_one_day_out_cross_validation(self):
        if self.save:
            self.experiment_folder = (
                f"{self.output_path}/{self.experiment_name + self._date_as_str()}"
            )
            self.create_output_folders()

        # Create folds
        kf = LeaveOneGroupOut()
        fold_models = []
        histories = []

        for fold_number, (train_indices, test_indices) in enumerate(
            kf.split(self.df, groups=self.df.date.dt.dayofweek)
        ):
            print(
                f"Fold ({fold_number + 1}/{len(np.unique(self.df.date.dt.dayofweek))}): "
            )

            train_df, test_df = self.df.iloc[train_indices], self.df.iloc[test_indices]
            train_df = self._drop_nan_rows(train_df)
            test_df = self._drop_nan_rows(test_df)

            fold_model, history = self.train_fold(fold_number, train_df, test_df)
            fold_models.append(fold_model)
            histories.append(history)

        return fold_models, histories

    def _drop_nan_rows(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        return df.dropna(subset=numeric_columns)

    def _split_train_test_by_stations(self, df, train_stations, test_stations):
        train_df = df[df[self.station_col].isin(train_stations)]
        test_df = df[df[self.station_col].isin(test_stations)]

        return train_df, test_df

    def split_train_test(self, train_stations, test_stations):
        train_df, test_df = self._split_train_test_by_stations(
            self.df, train_stations, test_stations
        )

        train_df = self._drop_nan_rows(train_df)
        test_df = self._drop_nan_rows(test_df)

        return train_df, test_df

    def create_features_target_sets(self, train_df, test_df):
        train_features = train_df[self.feature_groups.columns].to_numpy()
        test_features = test_df[self.feature_groups.columns].to_numpy()

        train_target = train_df[self.target_col].to_numpy()
        test_target = test_df[self.target_col].to_numpy()

        return train_features, test_features, train_target, test_target

    def preprocess_features(self, train_features, test_features):
        train_features = self.feature_preprocesses.fit_transform(train_features)
        test_features = self.feature_preprocesses.transform(test_features)

        return train_features, test_features

    def preprocess_images(self, train_images, test_images):
        for channel in np.arange(train_images.shape[-1]):
            images_mean, images_std = np.mean(
                train_images[:, :, :, channel].flatten()
            ), np.std(train_images[:, :, :, channel].flatten())
            train_images[:, :, :, channel] = (
                train_images[:, :, :, channel] - images_mean
            ) / images_std
            test_images[:, :, :, channel] = (
                test_images[:, :, :, channel] - images_mean
            ) / images_std

        return train_images, test_images

    def preprocess_context(self, train_context, test_context):
        train_context = self.context_preprocesses.fit_transform(
            train_context.reshape(-1, 1)
        )
        test_context = self.context_preprocesses.transform(test_context.reshape(-1, 1))

        return train_context, test_context

    def extract_buffers_distances_from_upstream_df(self, upstream_df):
        buffers, distances = np.stack(
            [
                feature.split("buffer")[1].split("_distance")
                for feature in upstream_df.columns.drop(["station", "date"]).to_list()
            ]
        ).swapaxes(0, 1)

        buffers = np.unique(buffers).astype(np.int32)
        distances = np.unique(distances).astype(np.int32)

        return np.sort(buffers), np.sort(distances)

    def upstream_features_to_images(self, buffers, distances, upstream_df):
        images = dict()

        for year in upstream_df.date.dt.year.unique():
            for idx, (station_idx, station_row) in enumerate(
                upstream_df.query("date.dt.year == @year").iterrows()
            ):
                n_rows = len(buffers)
                n_cols = len(distances)
                img = np.zeros((n_rows, n_cols, 1))

                for row, buffer in enumerate(buffers):
                    for col, distance in enumerate(distances):
                        for feature_idx, feature in enumerate(["mean"]):
                            img[row][col][feature_idx] = station_row[
                                f"{feature}_buffer{buffer}_distance{distance}"
                            ]

                images[(station_row["station"], year)] = img

        return images

    def images_dict_to_arrays(
        self,
        images_dict,
        buffers,
        distances,
        df,
    ):
        return np.stack(
            [
                images_dict[(row["station"], row["date"].year)]
                for _, row in df.iterrows()
            ]
        ).reshape(-1, len(buffers), len(distances), 1)

    def station_altitude_diff_to_arrays(self, df):
        return np.float32(
            [self.altitude_diff[row["station"]] for _, row in df.iterrows()]
        )

    def filter_missing_stations(self, df, images_dict):
        stations = list(np.unique([k[0] for k in list(images_dict.keys())]))
        return df.query("station in @stations")

    def prepare_data(self, train_df, test_df):
        # Shuffle train data
        train_df = train_df.sample(frac=1)

        # Filter stations that are not in images dict
        train_df = self.filter_missing_stations(
            train_df, self.upstream_slavi_images_dict
        )

        test_df = self.filter_missing_stations(test_df, self.upstream_slavi_images_dict)

        print("train_df size: ", len(train_df))
        print("test_df size: ", len(test_df))

        # Create SLAVI-VHM images arrays
        train_slavi_images = self.images_dict_to_arrays(
            self.upstream_slavi_images_dict,
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            train_df,
        )

        test_slavi_images = self.images_dict_to_arrays(
            self.upstream_slavi_images_dict,
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            test_df,
        )

        # Preprocess SLAVI-VHM images
        train_slavi_images, test_slavi_images = self.preprocess_images(
            train_slavi_images, test_slavi_images
        )

        # Create altitude diff arrays
        train_context = self.station_altitude_diff_to_arrays(train_df)
        test_context = self.station_altitude_diff_to_arrays(test_df)

        # Preprocess altitude diff arrays
        train_context, test_context = self.preprocess_context(
            train_context, test_context
        )

        # Create features & targets sets
        train_features, test_features, train_target, test_target = (
            self.create_features_target_sets(train_df, test_df)
        )

        # Preprocess features & targets sets
        train_features, test_features = self.preprocess_features(
            train_features, test_features
        )

        return {
            "train_features": train_features,
            "test_features": test_features,
            "train_target": train_target,
            "test_target": test_target,
            "train_slavi_images": train_slavi_images,
            "test_slavi_images": test_slavi_images,
            "train_context": train_context,
            "test_context": test_context,
        }

    def create_model_inputs(
        self,
        train_features,
        test_features,
        train_slavi_images,
        test_slavi_images,
        train_context,
        test_context,
    ):
        train_inputs, test_inputs = self.feature_groups.to_input_dictionaries(
            train_features, test_features
        )

        # Add images inputs
        train_inputs["slavi"] = train_slavi_images
        test_inputs["slavi"] = test_slavi_images

        # Add context inputs
        train_inputs["slope"] = train_context
        test_inputs["slope"] = test_context

        return train_inputs, test_inputs

    def train_fold(self, fold_number, train_df, test_df):
        data = self.prepare_data(train_df, test_df)

        train_features = data["train_features"]
        test_features = data["test_features"]
        train_target = data["train_target"]
        test_target = data["test_target"]
        train_slavi_images = data["train_slavi_images"]
        test_slavi_images = data["test_slavi_images"]
        train_context = data["train_context"]
        test_context = data["test_context"]

        train_inputs, test_inputs = self.create_model_inputs(
            train_features,
            test_features,
            train_slavi_images,
            test_slavi_images,
            train_context,
            test_context,
        )

        n_samples = 10  # FIXME: move to config
        min_alt = np.quantile(list(self.altitude_diff.values()), 0.25)
        max_alt = np.quantile(list(self.altitude_diff.values()), 0.75)

        tree_context_vis = np.round(
            np.linspace(min_alt, max_alt, n_samples).reshape(-1, 1), 1
        )
        tree_context_vis_preprocessed = self.context_preprocesses.transform(
            tree_context_vis
        )

        callbacks = [
            VisualizeVegetationWeightsCallback(
                train_tree=train_inputs["slavi"],
                context=tree_context_vis,
                context_preprocessed=tree_context_vis_preprocessed,
                distances=self.upstream_slavi_distances,
                buffers=self.upstream_slavi_buffers,
                n_samples=n_samples,
                experiment_folder=self.experiment_folder,
                fold_number=fold_number,
                epoch_to_plot=self.n_epochs - 1,
            ),
            VisualizeVegetationWeightsAtAltitudeCallback(
                train_tree=train_inputs["slavi"],
                context=tree_context_vis,
                context_preprocessed=tree_context_vis_preprocessed,
                distances=self.upstream_slavi_distances,
                buffers=self.upstream_slavi_buffers,
                n_samples=n_samples,
                experiment_folder=self.experiment_folder,
                fold_number=fold_number,
                epoch_to_plot=self.n_epochs - 1,
            ),
            VisualizeGroupCallback(
                self.feature_groups,
                experiment_folder=self.experiment_folder,
                fold_number=fold_number,
                epoch_to_plot=self.n_epochs - 1,
            ),
        ]

        # Build the model
        fold_model = self.model.create(
            self.feature_groups,
            self.upstream_slavi_buffers,
            self.upstream_slavi_distances,
            self.upstream_context_cols,
        )

        # Plot it
        if fold_number == 0:
            self.model.plot(fold_model)
            print(fold_model.summary())

        history = fold_model.fit(
            train_inputs,
            train_target,
            epochs=self.n_epochs,
            validation_data=(test_inputs, test_target),
            batch_size=self.batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        return fold_model, history
