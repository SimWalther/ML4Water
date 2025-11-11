import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


class Annigma:
    def __init__(self, model):
        """
        Annigma feature importance calculator for models with grouped inputs
        concatenated before hidden layers.

        Parameters:
            model: keras.Model
                A compiled Keras model with a structure including a Concatenate
                layer followed by Dense layers.
        """
        self.model = model
        self.annigma_feature_importance_dict = dict()

    def compute_from_concat_layer(self, concat_layer_name=None):
        """
        Calculates feature/group importances based on the concatenated representation
        before the hidden layer.

        Parameters:
            concat_layer_name: str, optional
                Name of the concatenation layer. If None, the first Concatenate
                layer is used.

        Returns:
            annigma: np.ndarray
                Array of importances (%), one per group/feature from the concatenated layer.
        """
        # Step 1: Find the concatenation layer
        concat_layer = None
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Concatenate) or (
                concat_layer_name and layer.name == concat_layer_name
            ):
                concat_layer = layer
                break

        if concat_layer is None:
            raise ValueError("Concatenation layer not found.")

        # Step 2: Find the first Dense layer after concatenation
        dense_hidden = None
        found_concat = False
        for layer in self.model.layers:
            if layer == concat_layer:
                found_concat = True
            elif found_concat and isinstance(layer, keras.layers.Dense):
                dense_hidden = layer
                break

        if dense_hidden is None:
            raise ValueError("Dense layer after concatenation not found.")

        # Step 3: Find the output Dense layer
        dense_output = None
        dense_found = False
        for layer in self.model.layers:
            if layer == dense_hidden:
                dense_found = True
            elif dense_found and isinstance(layer, keras.layers.Dense):
                dense_output = layer
                break

        if dense_output is None:
            raise ValueError("Output dense layer not found.")

        # Step 4: Extract weights
        W_hidden = dense_hidden.get_weights()
        W_out = dense_output.get_weights()
        if not W_hidden or not W_out:
            raise ValueError("Could not extract weights from Dense layers.")

        W_hidden = W_hidden[0]  # shape: (n_concat, n_neurons)
        W_out = W_out[0]        # shape: (n_neurons, 1)

        # Step 5: Compute contribution of each concatenated input
        linear_paths = np.abs(np.matmul(W_hidden, W_out))  # shape: (n_concat, 1)
        annigma = 100 * linear_paths / np.sum(linear_paths)

        return annigma.flatten()


    def compute_named_annigma(self, feature_groups):
        """
        Calculates and returns a sorted dict of feature/group importances (annigma), with proper names.
    
        Parameters:
            feature_groups: FeatureGroups
                An instance of FeatureGroups defining group memberships and types.
    
        Returns:
            sorted_annigma: dict
                Sorted dictionary {name: importance_percentage}, descending.
        """
        raw_annigma = self.compute_from_concat_layer()
    
        names = []
        for group_name, cols in feature_groups.groups.items():
            group_type = feature_groups.groups_types.get(group_name, "other")
            if group_type == "other":
                names.extend(cols)  # use actual feature names
            else:
                names.append(group_name)  # keep group name
        names.append('Group_vegetation')
        # Map annigma values back to names
        annigma_dict = dict(zip(names, raw_annigma))
        sorted_annigma = dict(
            sorted(annigma_dict.items(), key=lambda item: item[1], reverse=True)
        )
    
        return sorted_annigma


    def plot_single_annigma(self, annigma_dict, n_features=100, title_suffix="Model"):
        """
        Plots a horizontal bar chart of annigma values for a single model.

        Parameters:
            annigma_dict: dict or pd.Series
                Mapping feature_name -> importance
            n_features: int
                Number of top features to display
            title_suffix: str
                String to add to the plot title
        """
        # Convert to Series if necessary
        if isinstance(annigma_dict, dict):
            annigma_series = pd.Series(annigma_dict)
        else:
            annigma_series = annigma_dict

        # Sort features by importance
        annigma_sorted = annigma_series.sort_values()
        features_sorted = annigma_sorted.index.tolist()
        values_sorted = annigma_sorted.values

        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(
            range(len(values_sorted))[-n_features:],
            values_sorted[-n_features:],
            color="mediumseagreen",
        )
        plt.yticks(
            range(len(features_sorted))[-n_features:],
            features_sorted[-n_features:],
        )
        plt.xlabel("Feature Importance")
        plt.title(f"Annigma - {title_suffix}")
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_mean_annigma(self, n_features=100):
        """
        Plots the average annigma with standard deviation as error bars across folds.

        Parameters:
            n_features: int
                Number of top features to display
        """
        df = pd.DataFrame(self.annigma_feature_importance_dict).T
        means = df.mean()
        stds = df.std()

        sorted_indices = means.sort_values().index
        means_sorted = means[sorted_indices]
        stds_sorted = stds[sorted_indices]
        features_sorted = sorted_indices.tolist()

        if np.all(np.isnan(stds_sorted[-n_features:])):
            print("Standard deviations contain only NaN. Ignoring error bars.")
            xerr = None
        else:
            xerr = stds_sorted[-n_features:]

        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(means_sorted))[-n_features:],
            means_sorted[-n_features:],
            xerr=xerr,
            align="center",
            alpha=0.8,
            color="goldenrod",
            zorder=2,
        )
        plt.yticks(
            range(len(features_sorted))[-n_features:],
            features_sorted[-n_features:],
            rotation=0,
        )
        plt.ylabel("Features", fontsize=16)
        plt.xlabel("Average Feature Importance", fontsize=16)
        plt.title("Annigma - Mean Across Folds")
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.5, linewidth=1, zorder=1)
        plt.tight_layout()
        plt.show()
