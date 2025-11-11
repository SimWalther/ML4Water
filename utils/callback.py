from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class VisualizeVegetationWeightsCallback(Callback):
    def __init__(
        self,
        train_tree,
        context,
        context_preprocessed,
        buffers,
        distances,
        n_samples,
        experiment_folder,
        fold_number,
        epoch_to_plot,
    ):
        super(VisualizeVegetationWeightsCallback, self).__init__()
        self.n_samples = n_samples
        self.train_tree = train_tree
        self.context = context
        self.context_preprocessed = context_preprocessed
        self.buffers = buffers
        self.distances = distances
        self.experiment_folder = experiment_folder
        self.fold_number = fold_number
        self.epoch_to_plot = epoch_to_plot

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_to_plot:
            try:
                layer_name = "gaussianAttention2d"
                gaussian_attention_layer = self.model.get_layer(layer_name)

                # print(gaussian_attention_layer.get_weights())

                indices = np.random.choice(
                    len(self.train_tree), size=self.n_samples, replace=False
                )

                fig, (ax, ax2) = plt.subplots(
                    1,
                    2,
                    figsize=(10, 8),
                    layout="constrained",
                    sharey=True,
                    gridspec_kw={"width_ratios": [1, 6.25]},
                )

                # Plot upstream distance
                altitude_att_matrix = np.zeros((self.n_samples, len(self.distances)))

                for i, idx in enumerate(indices):
                    tree_batch = np.expand_dims(self.train_tree[idx], axis=0)
                    tree_batch = tf.cast(tree_batch, dtype="float32")

                    context_batch = np.expand_dims(self.context_preprocessed[i], axis=0)
                    context_batch = tf.cast(context_batch, dtype="float32")

                    attention_matrix = gaussian_attention_layer.attention_matrix(
                        tree_batch, context_batch
                    )
                    attention_matrix_img = np.squeeze(attention_matrix)
                    altitude_att_matrix[i] = np.sum(attention_matrix_img, axis=0)

                ax2.imshow(altitude_att_matrix)
                ax2.set_xlabel("Upstream distance [m]", fontsize=10)

                ax2.set_xticks(
                    np.arange(0, len(self.distances)), self.distances, rotation=90
                )
                ax2.set_yticks(
                    np.arange(0, self.n_samples),
                    np.concatenate(self.context, axis=None),
                )

                ax2.grid(True, linestyle="--", alpha=0.5, color="gray")

                np.save(
                    f"{self.experiment_folder}/upstream_attention/distance_attention_{self.fold_number}.npy",
                    attention_matrix_img,
                )

                # Plot buffer size
                altitude_att_matrix = np.zeros((self.n_samples, len(self.buffers)))

                for i, idx in enumerate(indices):
                    tree_batch = np.expand_dims(self.train_tree[idx], axis=0)
                    tree_batch = tf.cast(tree_batch, dtype="float32")

                    context_batch = np.expand_dims(self.context_preprocessed[i], axis=0)
                    context_batch = tf.cast(context_batch, dtype="float32")

                    attention_matrix = gaussian_attention_layer.attention_matrix(
                        tree_batch, context_batch
                    )
                    attention_matrix_img = np.squeeze(attention_matrix)

                    # Find the upstream distance with the highest values
                    altitude_att_matrix[i] = np.sum(attention_matrix_img, axis=1)

                np.save(
                    f"{self.experiment_folder}/upstream_attention/buffer_attention_{self.fold_number}.npy",
                    attention_matrix_img,
                )

                ax.imshow(altitude_att_matrix)
                ax.set_xlabel("Buffer width [m]", fontsize=10)
                ax.set_ylabel("Altitude difference [m]", fontsize=10)

                # ax2.set_ylabel("Altitude difference", fontsize=10)

                ax.set_xticks(
                    np.arange(0, len(self.buffers)), self.buffers, rotation=90
                )
                ax.set_yticks(
                    np.arange(0, self.n_samples),
                    np.concatenate(self.context, axis=None),
                )

                ax.grid(True, linestyle="--", alpha=0.5, color="gray")

                plt.show()

            except Exception as e:
                print(f"Error in visualization callback: {e}")
                print("Continuing training...")


class VisualizeVegetationWeightsAtAltitudeCallback(Callback):
    def __init__(
        self,
        train_tree,
        context,
        context_preprocessed,
        buffers,
        distances,
        n_samples,
        experiment_folder,
        fold_number,
        epoch_to_plot,
    ):
        super(VisualizeVegetationWeightsAtAltitudeCallback, self).__init__()
        self.n_samples = n_samples
        self.train_tree = train_tree
        self.context = context
        self.context_preprocessed = context_preprocessed
        self.buffers = buffers
        self.distances = distances
        self.experiment_folder = experiment_folder
        self.fold_number = fold_number
        self.epoch_to_plot = epoch_to_plot

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_to_plot:
            try:
                layer_name = "gaussianAttention2d"
                gaussian_attention_layer = self.model.get_layer(layer_name)

                fig, axs = plt.subplots(
                    2,
                    2,
                    layout="constrained",
                    figsize=(11, 3),
                    gridspec_kw={"width_ratios": [1, 25], "height_ratios": [4, 1]},
                )

                axs = axs.flatten()

                # Choose altitude difference at the middle
                middle_idx = len(self.context_preprocessed) // 2
                altitude_diff = self.context_preprocessed[middle_idx]
                altitude_diff_raw = self.context[middle_idx]

                # Plot upstream distance
                tree_batch = np.expand_dims(self.train_tree[0], axis=0)
                tree_batch = tf.cast(tree_batch, dtype="float32")

                context_batch = np.expand_dims(altitude_diff, axis=0)
                context_batch = tf.cast(context_batch, dtype="float32")

                attention_matrix = gaussian_attention_layer.attention_matrix(
                    tree_batch, context_batch
                )

                attention_matrix = np.squeeze(attention_matrix)

                fig.suptitle(
                    f"Gaussian Attention Matrix (altitude diff: {altitude_diff_raw[0]})"
                )
                axs[1].imshow(attention_matrix)
                axs[1].set_yticks(
                    np.arange(0, len(self.buffers)),
                    ["" for _ in np.arange(len(self.buffers))],
                )
                axs[1].set_xticks(
                    np.arange(0, len(self.distances)),
                    ["" for _ in np.arange(len(self.distances))],
                )
                axs[1].grid(True, linestyle="--", alpha=0.5, color="gray")

                # Plot buffer width
                attention_matrix_buffer = np.expand_dims(
                    np.sum(attention_matrix, axis=0), axis=0
                )
                axs[3].imshow(attention_matrix_buffer)
                axs[3].set_xlabel("Upstream Distance [m]")
                axs[3].grid(True, linestyle="--", alpha=0.5, color="gray")
                axs[3].set_yticks([])
                axs[3].set_xticks(
                    np.arange(0, len(self.distances)), self.distances, rotation=90
                )

                # Plot distance
                attention_matrix_distance = np.expand_dims(
                    np.sum(attention_matrix, axis=1), axis=-1
                )
                axs[0].imshow(attention_matrix_distance)
                axs[0].set_ylabel("Buffer Width [m]")
                axs[0].set_xticks([])
                axs[0].set_yticks(
                    np.arange(0, len(self.buffers)), self.buffers, rotation=90
                )
                axs[0].grid(True, linestyle="--", alpha=0.5, color="gray")

                fig.delaxes(axs[2])

                plt.show()
                plt.close()

            except Exception as e:
                print(f"Error in visualization callback: {e}")
                print("Continuing training...")


class VisualizeGroupCallback(Callback):
    def __init__(self, feature_groups, experiment_folder, fold_number, epoch_to_plot):
        super(VisualizeGroupCallback, self).__init__()
        self.feature_groups = feature_groups
        self.experiment_folder = experiment_folder
        self.fold_number = fold_number
        self.epoch_to_plot = epoch_to_plot

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_to_plot:
            try:
                num_groups = len(self.feature_groups.groups) - 1
                fig, axs = plt.subplots(
                    1,
                    num_groups,
                    figsize=(4 * num_groups, 2),
                    layout="constrained",
                )

                axs = axs.flatten()

                # plt.suptitle("Features Groups Weights")

                for group_idx, (feature_group, feature_names) in enumerate(
                    self.feature_groups.groups.items()
                ):
                    if feature_group == "input_other":
                        continue

                    attention_layer = self.model.get_layer(feature_group + "_attention")
                    attention_weights = tf.nn.softmax(
                        attention_layer.get_weights()
                    ).numpy()[0]

                    axs[group_idx].set_title(feature_group)
                    axs[group_idx].bar(feature_names, height=attention_weights)
                    axs[group_idx].tick_params(axis="x", labelrotation=90)

                    np.save(
                        f"{self.experiment_folder}/groups_attention/{feature_group}_{self.fold_number}.npy",
                        attention_weights,
                    )

                plt.show()
                plt.close()

            except Exception as e:
                print(f"Error in visualization callback: {e}")
                print("Continuing training...")
