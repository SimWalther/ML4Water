import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.ticker as mticker


def contains_kw(col, kw, all_features):
    # If kw is an exact feature name, require exact match
    if kw in all_features:
        return col == kw
    # Otherwise, regex match (kw as token in string)
    return re.search(rf"(?:^|_){re.escape(kw)}(?:_|$)", col) is not None


def calculate_simple_anigma(model_of_interest):
    w = model_of_interest.get_weights()
    lg = np.abs(np.matmul(w[0], w[2]))
    annigma = 100 * lg / np.sum(lg)
    return annigma.flatten()


def plot_absolute_errors_over_time(y_test, y_pred, test_days):
    # Calculate the errors
    errors = abs(y_test - y_pred)

    # Convert test_days to a format suitable for plotting (if they are not already in that format)
    test_days = pd.to_datetime(test_days)

    # Create the plot and get the Axes object
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        test_days,
        errors,
        linestyle="-",
        color="k",
        label="Absolute Error",
        linewidth=0.8,
    )

    # Set the x-ticks to the 1st of January of each year
    years = pd.date_range(start=test_days.min(), end=test_days.max(), freq="YS")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)

    # Format x-axis dates to "year-month-day"
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Set the labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Errors Over Time")

    # Define the colors for each season
    season_colors = {
        "Winter": "#B3DAF1",  # Light Blue
        "Spring": "#B4F9A5",  # Light Green
        "Summer": "#FFD700",  # Light Yellow
        "Fall": "#9C2706",  # Tan
    }

    # Add seasonal background colors
    for year in np.unique(test_days.year):
        winter_start = pd.Timestamp(year=year, month=12, day=1)
        winter_end = pd.Timestamp(year=year + 1, month=2, day=28)
        spring_start = pd.Timestamp(year=year, month=3, day=1)
        spring_end = pd.Timestamp(year=year, month=5, day=31)
        summer_start = pd.Timestamp(year=year, month=6, day=1)
        summer_end = pd.Timestamp(year=year, month=8, day=31)
        fall_start = pd.Timestamp(year=year, month=9, day=1)
        fall_end = pd.Timestamp(year=year, month=11, day=30)

        ax.axvspan(winter_start, winter_end, color=season_colors["Winter"], alpha=0.3)
        ax.axvspan(spring_start, spring_end, color=season_colors["Spring"], alpha=0.3)
        ax.axvspan(summer_start, summer_end, color=season_colors["Summer"], alpha=0.3)
        ax.axvspan(fall_start, fall_end, color=season_colors["Fall"], alpha=0.3)

    # Create proxy artists for the seasonal colors for the legend
    patches = [
        mpatches.Patch(color=color, label=season)
        for season, color in season_colors.items()
    ]

    # Add the legend
    ax.legend(
        handles=patches + [plt.Line2D([0], [0], color="r", label="Absolute Error")],
        bbox_to_anchor=(0.75, -0.25),
        shadow=True,
        ncol=5,
    )

    # Add grid lines for better readability
    ax.grid(True)

    plt.show()


def plot_anigma(N_features_to_plot=100):

    plt.figure(figsize=(12, 8))

    plt.barh(
        range(len(means_sorted))[-N_features_to_plot:],
        means_sorted[-N_features_to_plot:],
        xerr=stds_sorted[-N_features_to_plot:],
        align="center",
        alpha=0.8,
        color="goldenrod",
        zorder=2,
    )

    # Adjust the y-ticks to correspond to the last N features
    plt.yticks(
        range(len(features_sorted))[-N_features_to_plot:],
        features_sorted[-N_features_to_plot:],
        rotation=0,
    )

    # Add labels and title
    plt.ylabel("Features", fontsize=16)
    plt.xlabel("Average Feature Importance", fontsize=16)
    plt.title("Annigma")

    # Plot the grid
    plt.grid(alpha=0.5, linewidth=1, zorder=1)

    plt.xticks(np.arange(0, np.max(means_sorted) + 1, 1))

    # Adjust layout for better fit
    plt.tight_layout()

    plt.show()


def calculate_and_plot_anigma_for_this_station(N_features_to_plot=100):

    df_anigma = pd.DataFrame(
        Anigma_dict_by_fold[this_station][0],
        index=remaining_features,
        columns=[this_station],
    )
    means = df_anigma.mean(axis=1)
    stds = df_anigma.std(axis=1)

    # Zip the features, means, and stds together
    data_sorted = sorted(zip(remaining_features, means, stds), key=lambda x: x[1])

    # Extract sorted features, means, and stds
    features_sorted, means_sorted, stds_sorted = zip(*data_sorted)

    plt.figure(figsize=(12, 8))

    plt.barh(
        range(len(means_sorted))[-N_features_to_plot:],
        means_sorted[-N_features_to_plot:],
        xerr=stds_sorted[-N_features_to_plot:],
        align="center",
        alpha=0.8,
        color="goldenrod",
        zorder=2,
    )

    # Adjust the y-ticks to correspond to the last N features
    plt.yticks(
        range(len(features_sorted))[-N_features_to_plot:],
        features_sorted[-N_features_to_plot:],
        rotation=0,
    )

    # Add labels and title
    plt.ylabel("Features", fontsize=16)
    plt.xlabel("Average Feature Importance", fontsize=16)
    plt.title("Annigma")

    # Plot the grid
    plt.grid(alpha=0.5, linewidth=1, zorder=1)

    plt.xticks(np.arange(0, np.max(means_sorted) + 1, 1))

    # Adjust layout for better fit
    plt.tight_layout()

    plt.show()


def plot_loss_and_mae(history, fold_number=None):
    """
    Plot training and validation Loss & MAE with two y-axes.

    Parameters:
    - history: Keras History object (from model.fit)
    - fold_number: Optional fold number (int or str) for the title
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Loss on the left y-axis
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="blue")
    ax1.plot(history.history["loss"], label="Training Loss", color="blue")
    ax1.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="blue",
        linestyle="dashed",
    )
    ax1.tick_params(axis="y", labelcolor="blue")

    # # Plot MAE on the right y-axis
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('MAE', color='red')
    # ax2.plot(history.history['mae'], label='Training MAE', color='red')
    # ax2.plot(history.history['val_mae'], label='Validation MAE', color='red', linestyle='dashed')
    # ax2.tick_params(axis='y', labelcolor='red')

    # Title
    title = "Training and Validation Loss"
    if fold_number is not None:
        title += f" for Fold {fold_number}"
    plt.title(title)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    fig.legend(lines_1, labels_1, loc="upper right")

    plt.tight_layout()
    plt.show()


def calculate_anigma_from_concat_layer(model, concat_layer_name=None):
    """
    Calculates feature/group importances based on the concatenated representation before the hidden layer.

    Parameters:
        model: the Keras Model.
        concat_layer_name: optionally, the name of the concatenation layer.

    Returns:
        annigma: np.array of importances (%), one per group/feature from the concatenated layer.
    """
    # Step 1: Find the concatenation layer
    concat_layer = None
    for layer in model.layers:
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
    for layer in model.layers:
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
    for layer in model.layers:
        if layer == dense_hidden:
            dense_found = True
        elif dense_found and isinstance(layer, keras.layers.Dense):
            dense_output = layer
            break
    if dense_output is None:
        raise ValueError("Output dense layer not found.")

    # Step 4: Extract weights
    W_hidden = dense_hidden.get_weights()[0]  # shape: (n_concat, n_neurons)
    W_out = dense_output.get_weights()[0]  # shape: (n_neurons, 1)

    # Step 5: Compute contribution of each concatenated input
    linear_paths = np.abs(np.matmul(W_hidden, W_out))  # shape: (n_concat, 1)
    annigma = 100 * linear_paths / np.sum(linear_paths)

    return annigma.flatten()


def calculate_named_anigma(model, X_group_dict, group_definitions, other_feature_names):
    """
    Calculates and returns a sorted dict of feature/group importances (annigma), with proper names.

    Parameters:
        model: Trained Keras model.
        X_group_dict: dict of group_name -> np.array of inputs
        group_definitions: dict describing each group's type
        other_feature_names: list of strings, feature names for 'other' group

    Returns:
        Sorted dictionary of {name: importance_percentage}, descending.
    """
    raw_anigma = calculate_anigma_from_concat_layer(model)

    names = []
    other_index = 0

    for group_name, X_group in X_group_dict.items():
        group_type = group_definitions.get(group_name, {}).get("type", "other")
        if group_type == "other":
            n_features = X_group.shape[1]
            names.extend(other_feature_names[other_index : other_index + n_features])
            other_index += n_features
        else:
            names.append(group_name)

    # Create and sort the importance dictionary
    anigma_dict = dict(zip(names, raw_anigma))
    sorted_anigma = dict(
        sorted(anigma_dict.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_anigma


def plot_single_anigma(anigma_dict, N_features_to_plot=100, title_suffix="Model"):
    """
    Plots a horizontal bar chart of Anigma values for a single model.

    Parameters:
        anigma_dict: dict or pd.Series mapping feature_name -> importance
        N_features_to_plot: number of top features to display
        title_suffix: string to add to the plot title
    """
    # Convert to Series if necessary
    if isinstance(anigma_dict, dict):
        anigma_series = pd.Series(anigma_dict)
    else:
        anigma_series = anigma_dict

    # Sort features by importance
    anigma_sorted = anigma_series.sort_values()
    features_sorted = anigma_sorted.index.tolist()
    values_sorted = anigma_sorted.values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.barh(
        range(len(values_sorted))[-N_features_to_plot:],
        values_sorted[-N_features_to_plot:],
        color="mediumseagreen",
    )

    plt.yticks(
        range(len(features_sorted))[-N_features_to_plot:],
        features_sorted[-N_features_to_plot:],
    )

    plt.xlabel("Feature Importance")
    plt.title(f"Anigma - {title_suffix}")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_mean_anigma(Anigma_feature_importance_dict, N_features_to_plot=100):
    """
    Plots the average anigma with standard deviation as error bars across folds.

    Parameters:
        Anigma_feature_importance_dict: dict of fold_index -> {feature_name: importance}
        N_features_to_plot: number of top features to display
    """
    df = pd.DataFrame(Anigma_feature_importance_dict).T
    means = df.mean()
    stds = df.std()

    sorted_indices = means.sort_values().index
    means_sorted = means[sorted_indices]
    stds_sorted = stds[sorted_indices]
    features_sorted = sorted_indices.tolist()

    if np.all(np.isnan(stds_sorted[-N_features_to_plot:])):
        print("Standard deviations contain only NaN. Ignoring error bars.")
        xerr = None
    else:
        xerr = stds_sorted[-N_features_to_plot:]

    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(means_sorted))[-N_features_to_plot:],
        means_sorted[-N_features_to_plot:],
        xerr=xerr,
        align="center",
        alpha=0.8,
        color="goldenrod",
        zorder=2,
    )

    plt.yticks(
        range(len(features_sorted))[-N_features_to_plot:],
        features_sorted[-N_features_to_plot:],
        rotation=0,
    )

    plt.ylabel("Features", fontsize=16)
    plt.xlabel("Average Feature Importance", fontsize=16)
    plt.title("Annigma - Mean Across Folds")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.5, linewidth=1, zorder=1)
    plt.tight_layout()
    plt.show()


# Custom loss function with feature importance penalty
def custom_loss(y_true, y_pred):
    base_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)

    # Extract model weights from the first Dense layer
    weights = model.layers[1].kernel  # First Dense layer's weights

    # penalty = lambda_penalty * tf.reduce_sum(tf.abs(tf.gather(weights, non_important_feature_indices, axis=0)))
    penalty = lambda_penalty * tf.reduce_mean(
        tf.abs(tf.gather(weights, non_important_feature_indices, axis=0))
    )

    return base_loss + penalty  # Combined loss


def extract_first_number(s):
    """Extract the first occurring number from a string. Return a large value if none found."""
    match = re.search(r"\d+", s)
    return int(match.group()) if match else float("inf")


# Helper function to determine the season based on the date
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


# Helper function to determine the month based on the date
def get_month(date):
    return date.strftime("%B")


# Helper function to determine the hour based on the date
def get_hour(date):
    return date.hour


def compute_initial_weights(
    model, X_train_list, y_train_scaled, feature_groups, group_definitions
):
    """
    Compute initial weights for each feature group after one epoch with a single datapoint.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model.
    X_train_list : list of np.ndarray
        List of training inputs, one array per feature group.
    y_train_scaled : np.ndarray
        Target values (scaled).
    feature_groups : dict
        Dictionary mapping group names to feature indices.
    group_definitions : dict
        Dictionary defining group types, e.g. {'group_name': {'type': 'gaussian'|'softmax'|...}}.

    Returns
    -------
    dict
        Dictionary mapping each group to its initial weight vector.
    """

    # One epoch with a single datapoint to initialize group-specific weights
    model.fit([x[0:1] for x in X_train_list], y_train_scaled[0:1], epochs=1, verbose=0)

    initial_weights_by_group = {}

    for this_group in tqdm(feature_groups.keys(), desc="Extracting initial weights"):
        if this_group == "input_other":
            continue  # Skip direct-input features

        layer_type = group_definitions[this_group]["type"]

        if layer_type == "gaussian":
            this_layer = model.get_layer(name=f"gaussian_{this_group}")
            initial_weights = this_layer.final_weights.numpy().flatten()

        elif layer_type == "softmax":
            this_layer = model.get_layer(name=f"softmax_{this_group}")
            initial_weights = this_layer.get_weights()[
                0
            ].flatten()  # this_layer.get_weights_normalized().flatten()
            # Alternative (raw weights): this_layer.get_weights()[0].flatten()

        else:
            continue  # Skip unknown layer types

        initial_weights_by_group[this_group] = initial_weights

    return initial_weights_by_group


# def plot_group_weights(
#     model,
#     important_groups,
#     group_definitions,
#     sorted_features_by_group_dict,
#     initial_weights_by_group,
#     max_xticks=200
# ):
#     """
#     Plot initial vs final weights for groups (Gaussian or Softmax).

#     Args:
#         model: Trained Keras model.
#         important_groups: List of groups to plot.
#         group_definitions: Dict mapping group_name -> {'type': 'gaussian'/'softmax'/...}
#         sorted_features_by_group_dict: Dict mapping group_name -> ordered list of features.
#         initial_weights_by_group: Dict mapping group_name -> np.array of initial weights.
#         max_xticks: Maximum number of x-ticks to display on the plots.
#     """
#     for group_name in important_groups:
#         layer_type = group_definitions.get(group_name, {}).get("type", None)
#         if layer_type is None:
#             continue

#         sorted_features = sorted_features_by_group_dict[group_name]
#         initial_weights = initial_weights_by_group[group_name]

#         if layer_type == "gaussian":
#             layer = model.get_layer(name=f"gaussian_{group_name}")
#             final_weights = layer.final_weights.numpy().flatten()

#         elif layer_type == "softmax":
#             layer = model.get_layer(name=f"softmax_{group_name}")
#             # final_weights = layer.get_weights_normalized().flatten()
#             final_weights = layer.get_weights()[0].flatten()

#         else:
#             continue  # Skip unknown types

#         num_features = len(sorted_features)

#         # Choose xtick indices evenly spaced along the x-axis (up to max_xticks)
#         xtick_indices = np.linspace(0, num_features - 1, min(num_features, max_xticks), dtype=int)
#         xtick_labels = [sorted_features[i] for i in xtick_indices]

#         # --- Plot ---
#         plt.figure(figsize=(8, 2))
#         plt.plot(final_weights, label="Final Weights", color=plt.get_cmap("tab10")(1))
#         plt.plot(initial_weights, label="Initial Weights", linestyle="--", color=plt.get_cmap("tab10")(0))

#         plt.title(f"Weights evolution for {group_name} ({layer_type})")
#         plt.xlabel("Feature")
#         plt.ylabel("Weight")
#         plt.grid(True)

#         plt.xticks(ticks=xtick_indices, labels=xtick_labels, rotation=45, ha="right")
#         plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
#         plt.legend()
#         plt.show()


def plot_group_weights(
    model,
    important_groups,
    group_definitions,
    sorted_features_by_group_dict,
    initial_weights_by_group=None,
    max_xticks=200,
):
    """
    Plot initial vs final weights for groups (Gaussian or Softmax).

    Args:
        model: Trained Keras model.
        important_groups: List of groups to plot.
        group_definitions: Dict mapping group_name -> {'type': 'gaussian'/'softmax'/...}
        sorted_features_by_group_dict: Dict mapping group_name -> ordered list of features.
        initial_weights_by_group: (Optional) Dict mapping group_name -> np.array of initial weights.
                                  If None or missing for a group, only final weights are shown.
        max_xticks: Maximum number of x-ticks to display on the plots.
    """
    for group_name in important_groups:
        layer_type = group_definitions.get(group_name, {}).get("type", None)
        if layer_type is None:
            continue

        sorted_features = sorted_features_by_group_dict[group_name]
        num_features = len(sorted_features)

        # --- Get final weights from the model ---
        if layer_type == "gaussian":
            layer = model.get_layer(name=f"gaussian_{group_name}")
            final_weights = layer.final_weights.numpy().flatten()

        elif layer_type == "softmax":
            layer = model.get_layer(name=f"softmax_{group_name}")
            final_weights = layer.get_weights()[0].flatten()

        else:
            continue  # Skip unknown types

        # --- Get initial weights if available ---
        initial_weights = None
        if initial_weights_by_group is not None:
            initial_weights = initial_weights_by_group.get(group_name, None)

        # --- Choose xtick indices evenly spaced ---
        xtick_indices = np.linspace(
            0, num_features - 1, min(num_features, max_xticks), dtype=int
        )
        xtick_labels = [sorted_features[i] for i in xtick_indices]

        # --- Plot ---
        plt.figure(figsize=(8, 2))
        plt.plot(final_weights, label="Final Weights", color=plt.get_cmap("tab10")(1))

        if initial_weights is not None:
            plt.plot(
                initial_weights,
                label="Initial Weights",
                linestyle="--",
                color=plt.get_cmap("tab10")(0),
            )

        plt.title(f"Weights evolution for {group_name} ({layer_type})")
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.grid(True)

        plt.xticks(ticks=xtick_indices, labels=xtick_labels, rotation=45, ha="right")
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        plt.legend()
        plt.show()


def plot_predictions_over_time(y_test, y_pred, test_days, station_name):

    # Convert test_days to a format suitable for plotting (if they are not already in that format)
    test_days = pd.to_datetime(test_days)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ax.plot(test_days, y_test, label='Actual Water Temperature',  linestyle='-', color='blue')
    ax.plot(
        test_days,
        y_pred,
        label="Predicted Water Temperature",
        linestyle="-",
        color="cyan",
    )

    # Set the x-ticks to the 1st of January of each year
    years = pd.date_range(start=test_days.min(), end=test_days.max(), freq="YS")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)

    # Format x-axis dates to "year-month-day"
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Set the labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Temperature (°C)")
    ax.set_title(f"Predictions for {station_name}")

    ax.legend()
    ax.grid()
    # fig.autofmt_xdate()  # Automatically formats x-axis dates for better readability

    plt.show()
