import numpy as np

class Annigma:
    def __init__(self):
        pass

    def compute_simple(self, model):
        w = model.get_weights()
        lg = np.abs(np.matmul(w[0], w[2]))
        annigma = (100 * lg / np.sum(lg))
        return annigma.flatten()

    def plot(self, n_features = 100):
        plt.figure(figsize=(12, 8))
        
        plt.barh(range(len(means_sorted))[-n_features:], 
                means_sorted[-n_features:], 
                xerr=stds_sorted[-n_features:], 
                align='center', 
                alpha=0.8, 
                color='goldenrod', 
                zorder=2)
        
        # Adjust the y-ticks to correspond to the last N features
        plt.yticks(range(len(features_sorted))[-n_features:], 
                features_sorted[-n_features:], 
                rotation=0)
        
        # Add labels and title
        plt.ylabel('Features', fontsize=16)
        plt.xlabel('Average Feature Importance', fontsize=16)
        plt.title('Annigma')

        # Plot the grid
        plt.grid(alpha=0.5, linewidth=1, zorder=1)
        
        plt.xticks(np.arange(0, np.max(means_sorted) + 1, 1))
        
        # Adjust layout for better fit
        plt.tight_layout()
        plt.show()


def compute_and_plot(selectedn_features = 100):
    df_anigma = pd.DataFrame(Anigma_dict_by_fold[this_station][0], index=remaining_features,columns=[this_station])
    means = df_anigma.mean(axis=1)
    stds = df_anigma.std(axis=1)
    
    # Zip the features, means, and stds together
    data_sorted = sorted(zip(remaining_features, means, stds), key=lambda x: x[1])
    
    # Extract sorted features, means, and stds
    features_sorted, means_sorted, stds_sorted = zip(*data_sorted)

    plt.figure(figsize=(12, 8))
    
    plt.barh(range(len(means_sorted))[-N_features_to_plot:], 
             means_sorted[-N_features_to_plot:], 
             xerr=stds_sorted[-N_features_to_plot:], 
             align='center', 
             alpha=0.8, 
             color='goldenrod', 
             zorder=2)
    
    # Adjust the y-ticks to correspond to the last N features
    plt.yticks(range(len(features_sorted))[-N_features_to_plot:], 
               features_sorted[-N_features_to_plot:], 
               rotation=0)
    
    # Add labels and title
    plt.ylabel('Features', fontsize=16)
    plt.xlabel('Average Feature Importance', fontsize=16)
    plt.title('Annigma')

    # Plot the grid
    plt.grid(alpha=0.5, linewidth=1, zorder=1)
    
    plt.xticks(np.arange(0, np.max(means_sorted) + 1, 1))
    
    # Adjust layout for better fit
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
        if isinstance(layer, keras.layers.Concatenate) or (concat_layer_name and layer.name == concat_layer_name):
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
    W_out = dense_output.get_weights()[0]     # shape: (n_neurons, 1)

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
            names.extend(other_feature_names[other_index:other_index + n_features])
            other_index += n_features
        else:
            names.append(group_name)

    # Create and sort the importance dictionary
    anigma_dict = dict(zip(names, raw_anigma))
    sorted_anigma = dict(sorted(anigma_dict.items(), key=lambda item: item[1], reverse=True))

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
    plt.barh(range(len(values_sorted))[-N_features_to_plot:], 
             values_sorted[-N_features_to_plot:], 
             color='mediumseagreen')

    plt.yticks(range(len(features_sorted))[-N_features_to_plot:], 
               features_sorted[-N_features_to_plot:])
    
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
    plt.barh(range(len(means_sorted))[-N_features_to_plot:], 
             means_sorted[-N_features_to_plot:], 
             xerr=xerr, 
             align='center', 
             alpha=0.8, 
             color='goldenrod', 
             zorder=2)

    plt.yticks(range(len(features_sorted))[-N_features_to_plot:], 
               features_sorted[-N_features_to_plot:], 
               rotation=0)

    plt.ylabel('Features', fontsize=16)
    plt.xlabel('Average Feature Importance', fontsize=16)
    plt.title('Annigma - Mean Across Folds')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.5, linewidth=1, zorder=1)
    plt.tight_layout()
    plt.show()    