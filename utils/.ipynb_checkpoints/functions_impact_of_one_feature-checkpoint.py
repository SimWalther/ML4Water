from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

import sys

sys.path.append(os.path.abspath('/home/benoit.hohl/shared-projects/ML4WATER/Uniformise/utils'))

import utils_functions_benoit
from group_definitions import group_definitions


def prepare_data(df, test_station, feature_of_interest=None, fixed_value=None):
    """
    Split data into train and test, optionally modify a feature in test set.
    Returns: X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_test_original, test_days
    """
    df_train = df[~(df['station'] == test_station)].copy()
    df_test = df[df['station'] == test_station].copy()

    numeric_columns = df_train.select_dtypes(include=[np.number]).columns
    df_train = df_train.dropna(subset=numeric_columns)
    df_test = df_test.dropna(subset=numeric_columns)

    if feature_of_interest and fixed_value is not None:
        df_test[feature_of_interest] = fixed_value

    drop_cols = ['station', 'river_name', 'date']
    df_train_model = df_train.drop(columns=drop_cols)
    df_test_model = df_test.drop(columns=drop_cols)

    X_train = df_train_model.drop(columns=['water_temperature'])
    y_train = df_train_model['water_temperature']
    X_test = df_test_model.drop(columns=['water_temperature'])
    y_test = np.array(df_test_model['water_temperature']).flatten()
    test_days = np.array(df_test['date']).flatten()

    # Scale features and target
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_test, test_days, x_scaler, y_scaler

def evaluate_model(model, X_test_scaled, y_scaler):
    """
    Predict with model and return scaled and inverse-scaled predictions.
    """
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    return y_pred

def compare_feature_modification_effect(df, test_station, model, feature_of_interest, fixed_value, features_by_group = False):
    """
    Compare predictions with original vs modified feature.
    Returns dictionaries of results.
    """
    results_original = {}
    results_modified = {}

    for mode in ["original", "modified"]:
        modify = feature_of_interest if mode == "modified" else None


        if features_by_group:
            _, X_list,_, _,_,_, _ ,y, days,_, y_scaler,_= prepare_grouped_data(
        df,test_station,group_definitions, feature_of_interest=modify, fixed_value=fixed_value)
            X_input = X_list
        else:
            _, X, _ , _, y, days, _, y_scaler = prepare_data(
                df, test_station, feature_of_interest=modify, fixed_value=fixed_value)    
            X_input = X
        

        y_pred = evaluate_model(model, X_input, y_scaler)
        mae = mean_absolute_error(y, y_pred)

        # Store results directly in the dictionary
        output_dict = {
            'y': y,
            'y_pred': y_pred,
            'days': days,
            'mae': mae
        }

        if mode == "original":
            results_original = output_dict
        else:
            results_modified = output_dict

    return results_original, results_modified


def compute_output_and_error_of_model(X,y,days,y_scaler, model):
    """
    Compare predictions with original vs modified feature.
    Returns dictionaries of results.
    """
    results = {}
    y_pred = evaluate_model(model, X, y_scaler)
    mae = mean_absolute_error(y, y_pred)

    # Store results directly in the dictionary
    results = {
        'y': y,
        'y_pred': y_pred,
        'days': days,
        'mae': mae
    }
    return results


def compare_results(results_original, results_modified):
    """
    Compare original vs modified feature results for a single station.
    Expects results_original and results_modified to be dictionaries with keys:
    'y', 'y_pred', 'days', 'mae'
    """
    mae_orig = results_original['mae']
    mae_mod = results_modified['mae']

    print("\n--- Comparison ---")
    print(f"Original MAE: {mae_orig}")
    print(f"Modified MAE: {mae_mod}")
    print(f"Difference: {mae_mod - mae_orig}")




def plot_comparison_predictions(y, y_pred_original, y_pred_modified, days, station_name):

    # Convert days to a format suitable for plotting (if they are not already in that format)
    days = pd.to_datetime(days) 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    # ax.plot(days, y, label='Actual Water Temperature', linestyle='-', color='blue')
    
    # Plot original predictions
    ax.plot(days, y_pred_original, label='Predicted (Original)', linestyle='-', color='cyan')
    
    # Plot modified predictions
    ax.plot(days, y_pred_modified, label='Predicted (Feature modified)', linestyle='--', color='red')

    # Set the x-ticks to the 1st of January of each year
    years = pd.date_range(start=days.min(), end=days.max(), freq='YS')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)

    # Format x-axis dates to "year-month-day"
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Temperature (°C)')
    ax.set_title(f'Comparison of Predictions for {station_name}')
    
    ax.legend()
    ax.grid()

    plt.show()



def feature_sensitivity_for_one_day(
    model,
    final_df,
    station_name,
    target_day,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    features_by_group=False,
    feature_groups=None
):
    """
    Analyzes the impact of a feature on the prediction for a specific day by varying it from vmin to vmax.

    X_input has to be created based on the model used
    """

    # Select the specific datapoint
    selected_row = final_df[(final_df['station'] == station_name) & (final_df['date'] == target_day)].copy()

    if selected_row.empty:
        print("No matching row found for the given station and day.")
        return

    # Get the original feature value
    original_value = selected_row[feature_of_interest].values[0]

    # Drop non-feature columns
    X_fixed = selected_row.drop(columns=['station', 'river_name', 'date', 'water_temperature'])

    # Generate values for the feature_of_interest
    feature_values = np.linspace(vmin, vmax, num_points)
    predictions = []

    for value in feature_values:
        X_temp = X_fixed.copy()
        X_temp[feature_of_interest] = value  # Modify the feature

        if features_by_group:
            if feature_groups is None:
                raise ValueError("feature_groups must be provided when features_by_group=True")

            X_scaled_df = pd.DataFrame(x_scaler.transform(X_temp), columns=X_temp.columns)
            # X_input = [X_scaled_df[cols].values for cols in feature_groups.values()]
            X_input, _, _ = prepare_grouped_inputs(X_scaled_df, feature_groups)
        else:
            X_scaled = x_scaler.transform(X_temp)
            X_input = X_scaled

        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        predictions.append(y_pred[0])

    
    if features_by_group:
        X_scaled_df = pd.DataFrame(x_scaler.transform(X_fixed), columns=X_fixed.columns)
        # X_input = [X_scaled_df[cols].values for cols in feature_groups.values()]
        X_input, _, _ = prepare_grouped_inputs(X_scaled_df, feature_groups)
    else:
        X_scaled = x_scaler.transform(X_fixed)
        X_input = X_scaled

    y_orig_scaled = model.predict(X_input, verbose=0)
    y_orig = y_scaler.inverse_transform(y_orig_scaled).flatten()[0]

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(feature_values, predictions, marker='o', linestyle='-', color='purple', label='Predicted Temperature')

    # Highlight the original feature value with its actual prediction
    plt.scatter(original_value, y_orig, color='red', zorder=3, label='Original Prediction')

    plt.annotate(f'Original: {original_value:.2f}', 
                 xy=(original_value, np.interp(original_value, feature_values, predictions)), 
                 xytext=(original_value + (vmax - vmin) * 0.05, min(predictions)), 
                 arrowprops=dict(facecolor='red', arrowstyle='->'))

    plt.xlabel(f'{feature_of_interest} Value')
    plt.ylabel('Predicted Water Temperature (°C)')
    plt.title(f'Impact of {feature_of_interest} on Prediction\nStation: {station_name}, Date: {target_day}')
    plt.legend()
    plt.grid()
    plt.show()

    return feature_values, predictions, original_value, y_orig




def get_sorted_affected_days(results_original, results_modified):
    """
    Returns all days sorted by the absolute difference between y_pred_original and y_pred_modified.

    Parameters:
        results_original: Dict containing original predictions
        results_modified: Dict containing modified predictions


    Returns:
        sorted_df: DataFrame with days sorted by difference (descending)
    """
    y_pred_original = np.array(results_original['y_pred'])
    y_pred_modified = np.array(results_modified['y_pred'])
    days = np.array(results_original['days'])

    # Compute differences
    # differences = np.abs(y_pred_original - y_pred_modified)
    differences = (y_pred_original - y_pred_modified)

    # Create a DataFrame and sort by difference
    sorted_df = pd.DataFrame({'date': days, 'difference': differences})
    sorted_df = sorted_df.sort_values(by='difference', ascending=False).reset_index(drop=True)

    return sorted_df




def prepare_grouped_inputs(X_scaled_df, feature_groups):
    """
    Given a scaled DataFrame X_scaled_df and feature_groups,
    return (X_list, X_dict, sorted_features_by_group_dict).

    Args:
        X_scaled_df (pd.DataFrame): Scaled input features.
        feature_groups (dict): Dict {group_name: [feature1, feature2, ...]}.

    Returns:
        X_list (list of np.ndarray): List of arrays per group.
        X_dict (dict): Dict {group_name: array}.
        sorted_features_by_group_dict (dict): Dict {group_name: sorted feature list}.
    """
    X_list = []
    X_dict = {}
    sorted_features_by_group_dict = {}

    for this_group, features_temp in feature_groups.items():
        # Sort features by first number in their name (if any)
        sorted_features = sorted(features_temp, key=utils_functions_benoit.extract_first_number)
        sorted_features_by_group_dict[this_group] = sorted_features

        df_features = X_scaled_df[sorted_features]

        X_list.append(df_features.to_numpy())
        X_dict[this_group] = df_features.to_numpy()

    return X_list, X_dict, sorted_features_by_group_dict


def create_feature_groups(df, group_definitions):
    """
    Create feature groups from group_definitions.

    Args:
        df (pd.DataFrame): Input DataFrame with all columns (including station, date, etc.).
        group_definitions (dict): Dict {group_name: {"necessary": [...], "unwanted": [...]}}

    Returns:
        feature_groups (dict): Dict {group_name: [feature1, feature2, ...], "input_other": [...]}
    """
    # Columns that are NOT features
    columns_to_drop_for_model = ['station','river_name','date','water_temperature']
    all_features = df.drop(columns=columns_to_drop_for_model, axis=1).columns

    feature_groups = {}
    for group_name, rules in group_definitions.items():
        necessary_keywords = rules["necessary"]
        unwanted_keywords = rules["unwanted"]

        matched_cols_temp = [
            col for col in all_features
            if any(utils_functions_benoit.contains_kw(col, kw, all_features) for kw in necessary_keywords)
        ]
        matched_cols = [
            col for col in matched_cols_temp
            if not any(utils_functions_benoit.contains_kw(col, kw, all_features) for kw in unwanted_keywords)
        ]
        feature_groups[group_name] = matched_cols

    # Add "input_other" = columns not in any group
    grouped_columns = set(col for cols in feature_groups.values() for col in cols)
    feature_groups["input_other"] = [col for col in all_features if col not in grouped_columns]

    return feature_groups

def prepare_grouped_data(
    df,
    test_station,
    group_definitions,
    feature_of_interest=None,
    fixed_value=None,
    x_scaler=None,
    y_scaler=None
):
    """
    Split data into train and test (based on test_station),
    build feature groups from group_definitions,
    optionally modify a feature in the test set,
    prepare grouped feature arrays for the custom model,
    optionally use provided scalers for X and y.
    Returns None if train or test set is empty.
    """
    # --- Train/test split ---
    df_train = df[~(df['station'] == test_station)].copy()
    df_test = df[df['station'] == test_station].copy()

    # Keep numeric columns only & drop rows with NaNs
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns
    df_train = df_train.dropna(subset=numeric_columns)

    # print("Before dropna:", df_test["date"].min(), df_test["date"].max(), df_test.shape)
    df_test = df_test.dropna(subset=numeric_columns)
    # print("After dropna:", df_test["date"].min(), df_test["date"].max(), df_test.shape)


    # If train or test is empty, return None
    if df_train.shape[0] == 0 or df_test.shape[0] == 0:
        print(f"Warning: Station '{test_station}' has empty train or test set. Skipping.")
        return None

    # Optional modification in test set
    if feature_of_interest and fixed_value is not None:
        if feature_of_interest in df_test.columns:
            df_test[feature_of_interest] = fixed_value
        else:
            raise ValueError(f"Feature '{feature_of_interest}' not found in DataFrame.")

    # --- Create feature groups ---
    feature_groups = create_feature_groups(df, group_definitions)

    # --- Drop non-model cols ---
    drop_cols = ['station','river_name','date']
    df_train_model = df_train.drop(columns=drop_cols)
    df_test_model = df_test.drop(columns=drop_cols)

    # --- Split features and targets ---
    X_train = df_train_model.drop(columns=['water_temperature'])
    y_train = df_train_model['water_temperature'].to_numpy().flatten()
    X_test = df_test_model.drop(columns=['water_temperature'])
    y_test_original = df_test_model['water_temperature'].to_numpy().flatten()
    test_days = df_test['date'].to_numpy().flatten()

    # If y_train or y_test is empty, return None
    if len(y_train) == 0 or len(y_test_original) == 0:
        print(f"Warning: Station '{test_station}' has empty target after processing. Skipping.")
        return None

    # --- Scale target ---
    if y_scaler is None:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    else:
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test_original.reshape(-1, 1))

    # --- Scale features ---
    if x_scaler is None:
        x_scaler = StandardScaler()
        X_train_scaled_df = pd.DataFrame(
            x_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    else:
        X_train_scaled_df = pd.DataFrame(
            x_scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    X_test_scaled_df = pd.DataFrame(
        x_scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # --- Create grouped inputs ---
    X_train_list, X_train_dict, sorted_features_train = prepare_grouped_inputs(X_train_scaled_df, feature_groups)
    X_test_list, X_test_dict, sorted_features_test = prepare_grouped_inputs(X_test_scaled_df, feature_groups)

    # sanity check: sorting should be the same for train/test
    assert sorted_features_train == sorted_features_test, "Mismatch in feature sorting!"
    sorted_features_by_group_dict = sorted_features_train

    return (
        X_train_list, X_test_list,
        X_train_dict, X_test_dict,
        sorted_features_by_group_dict,
        y_train_scaled, y_test_scaled,
        y_test_original, test_days,
        x_scaler, y_scaler,
        feature_groups
    )


def filter_days(
    X_list,
    days,
    days_to_keep
):
    """
    Keep only the samples corresponding to specific days.

    Parameters:
        X_list: list of numpy arrays (grouped features)
        days: np.ndarray of  dates
        days_to_keep: list of days (as strings 'YYYY-MM-DD' or np.datetime64) to keep

    Returns:
        Filtered versions of:
        - X_list_filtered
        - days_filtered
    """
    # ensure days are comparable
    days_to_keep = np.array(days_to_keep, dtype=days.dtype)

    # Boolean mask
    mask = np.isin(days, days_to_keep)

    # Filter arrays
    X_list_filtered = [X[mask] for X in X_list]
    days_filtered = days[mask]

    return (
        X_list_filtered,
        days_filtered
    )

def get_hot_days(df, threshold_Ta):
    """
    Return list of dates where air_temperature > threshold_Ta.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'air_temperature' and 'date' columns.
    threshold_Ta : float
        Temperature threshold.
    
    Returns
    -------
    list of str
        Dates (as strings) where air_temperature > threshold_Ta.
    """
    mask = df["air_temperature"] > threshold_Ta
    hot_dates = df.loc[mask, "date"].unique().tolist()
    return sorted(hot_dates)


def get_cold_days(df, threshold_Ta):
    """
    Return list of dates where air_temperature > threshold_Ta.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'air_temperature' and 'date' columns.
    threshold_Ta : float
        Temperature threshold.
    
    Returns
    -------
    list of str
        Dates (as strings) where air_temperature > threshold_Ta.
    """
    mask = df["air_temperature"] < threshold_Ta
    hot_dates = df.loc[mask, "date"].unique().tolist()
    return sorted(hot_dates)

import pandas as pd

def get_days_between_md_in_df(df, start_md, end_md):
    """
    Return a list of dates from df['date'] that fall between start_md and end_md
    (inclusive), ignoring the year.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'date' column (as datetime or string).
    start_md : str
        Start date as 'MM-DD'.
    end_md : str
        End date as 'MM-DD'.

    Returns
    -------
    list of str
        Dates from df['date'] in 'YYYY-MM-DD' format.
    """
    # Ensure 'date' column is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Extract month-day as strings
    df['md'] = df['date'].dt.strftime('%m-%d')

    # Handle year-wrap (e.g., Dec -> Jan)
    if start_md <= end_md:
        mask = (df['md'] >= start_md) & (df['md'] <= end_md)
    else:
        # Wrap around year end
        mask = (df['md'] >= start_md) | (df['md'] <= end_md)

    return df.loc[mask, 'date'].dt.strftime('%Y-%m-%d').tolist()



def feature_sensitivity_for_days(
    model,
    final_df,
    station_name,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    feature_groups=None,
    ax=None,
    plot = True
):
    """
    Analyzes the impact of a feature on the prediction for multiple days
    by varying it from vmin to vmax and aggregating mean/std predictions.
    If `ax` is provided, plot into it. Otherwise, create a new figure.
    """

    if isinstance(target_days, str):  # allow single day string
        target_days = [target_days]

    selected_rows = final_df[(final_df['station'] == station_name) & 
                             (final_df['date'].isin(target_days))].copy()

    if selected_rows.empty:
        print(f"No matching rows found for station {station_name}.")
        return None

    X_fixed_all = selected_rows.drop(columns=['station', 'river_name', 'date', 'water_temperature'])
    original_values = selected_rows[feature_of_interest].values
    original_predictions = []

    feature_values = np.linspace(vmin, vmax, num_points)
    all_predictions = []

    for value in feature_values:

        X_temp = X_fixed_all.copy()
        X_temp[feature_of_interest] = value  # same feature value for all rows
        
        # Scale all rows at once
        X_scaled_df = pd.DataFrame(x_scaler.transform(X_temp), columns=X_temp.columns)
        X_input, _, _ = prepare_grouped_inputs(X_scaled_df, feature_groups)
        
        # Predict for all rows simultaneously
        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

        preds_for_value = y_pred.tolist()

        all_predictions.append(preds_for_value)

    all_predictions = np.array(all_predictions)
    mean_predictions = np.nanmean(all_predictions, axis=1)
    std_predictions = np.nanstd(all_predictions, axis=1)

    # for i in range(len(X_fixed_all)):
    #     X_fixed = X_fixed_all.iloc[[i]].copy()

    #     X_scaled_df = pd.DataFrame(x_scaler.transform(X_fixed), columns=X_fixed.columns)
    #     X_input, _, _ = prepare_grouped_inputs(X_scaled_df, feature_groups)

    #     y_orig_scaled = model.predict(X_input, verbose=0)
    #     y_orig = y_scaler.inverse_transform(y_orig_scaled).flatten()[0]
    #     original_predictions.append(y_orig)

    if plot:
        # === Plotting ===
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(feature_values, mean_predictions, color='purple', label='Mean Prediction')
        ax.fill_between(feature_values,
                        mean_predictions - std_predictions,
                        mean_predictions + std_predictions,
                        color='purple', alpha=0.2, label='± Std Dev')
        # ax.scatter(original_values, original_predictions, color='k', marker="x", s=10, label='Original Predictions')
        ax.axvline(x=original_values[0], color='k', ls='--', label='Original Value')
    
        ax.set_title(f'Station: {station_name}, Days: {len(target_days)}')
        ax.grid(True)
    
        if ax is None:
            ax.set_ylabel('Predicted Water Temperature (°C)')
            ax.set_xlabel(f'{feature_of_interest} Value')
            ax.legend()
            plt.show()

    return feature_values, mean_predictions, std_predictions, original_values





def plot_feature_sensitivity_multiple_stations(
    model,
    final_df,
    station_names,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    features_by_group=False,
    feature_groups=None,
    ncols=2,
    title=None
):
    """
    Creates a figure with subplots for multiple stations, using
    feature_sensitivity_for_days for each station.
    """

    nstations = len(station_names)
    nrows = int(np.ceil(nstations / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        sharex=True, sharey=False,
    )
    axes = np.array(axes).reshape(-1)

    results = {}
    handles, labels = None, None  # for legend collection

    for i, station in enumerate(station_names):
        ax = axes[i]

        res = feature_sensitivity_for_days(
            model=model,
            final_df=final_df,
            station_name=station,
            target_days=target_days,
            feature_of_interest=feature_of_interest,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            vmin=vmin,
            vmax=vmax,
            num_points=num_points,
            feature_groups=feature_groups,
            ax=ax  # <- plot into existing subplot
        )

        if res is None:
            ax.set_title(f"{station}: No data")
            continue

        results[station] = res

        # grab legend handles only once (from first subplot with data)
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()

    # Remove unused axes if grid > stations
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add one shared legend on top
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=3,fontsize=20,bbox_to_anchor=(0.5, 1.0))

    # Shared labels
    # fig.supxlabel(f"{feature_of_interest} Value")
    # fig.supylabel("Predicted Water Temperature (°C)")
    fig.text(0.5, -0.01, f"{feature_of_interest} Value", ha="center", fontsize=24)
    fig.text(-0.01, 0.5, "Predicted Water Temperature (°C)", va="center", rotation="vertical", fontsize=24)


    # Global title below the legend
    if title:
        fig.suptitle(title, fontsize=50, y=1.03)

    # leave enough space for both legend + title
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()

    return results


def plot_feature_sensitivity_all_stations(
    model,
    final_df,
    station_names,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    feature_groups=None,
    title=None,
    figsize=(10, 6)
):
    """
    Creates one plot where all stations are shown together,
    each with its feature sensitivity curve (no std band, no original values).
    """
    fig, ax = plt.subplots(figsize=figsize)

    results = {}

    for station in station_names:
        res = feature_sensitivity_for_days(
            model=model,
            final_df=final_df,
            station_name=station,
            target_days=target_days,
            feature_of_interest=feature_of_interest,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            vmin=vmin,
            vmax=vmax,
            num_points=num_points,
            feature_groups=feature_groups,
            ax=None,
            plot=False
        )

        if res is None:
            print(f"Skipping {station}: no data")
            continue

        feature_values, mean_predictions, _, _ = res
        results[station] = res

        # Plot each station’s curve
        ax.plot(feature_values, mean_predictions, label=f"{station}")

    # Labels, legend, grid
    ax.set_xlabel(f"{feature_of_interest} Value", fontsize=14)
    ax.set_ylabel("Predicted Water Temperature (°C)", fontsize=14)
    ax.grid(True)
    
    # Legend outside on the right
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=10,
        ncols=3
    )

    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    return results



def plot_feature_sensitivity_mean_across_stations(
    model,
    final_df,
    station_names,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    feature_groups=None,
    title=None,
    figsize=(10, 6)
):
    """
    Creates one plot showing the mean sensitivity curve
    across all stations with shaded interquartile range (25-75%).
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_curves = []
    feature_values = None

    n_stations=len(station_names)
    for station in station_names:
        res = feature_sensitivity_for_days(
            model=model,
            final_df=final_df,
            station_name=station,
            target_days=target_days,
            feature_of_interest=feature_of_interest,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            vmin=vmin,
            vmax=vmax,
            num_points=num_points,
            feature_groups=feature_groups,
            ax=None,
            plot=False
        )

        if res is None:
            print(f"Skipping {station}: no data")
            continue

        feature_values, mean_predictions, _, _ = res
        all_curves.append(mean_predictions)

    if not all_curves:
        print("No data for any station.")
        return None

    all_curves = np.array(all_curves)  # shape = (n_stations, num_points)

    # Median curve across stations
    mean_curve = np.nanmean(all_curves, axis=0)
    # Interquartile range (25th to 75th percentile)
    
    std = np.nanstd(all_curves, axis=0)
    
    # q25 = np.percentile(all_curves, 25, axis=0)
    # q75 = np.percentile(all_curves, 75, axis=0)

    # Plot mean
    ax.plot(feature_values, mean_curve, color="darkorange", linewidth=2, label=f"Mean across stations (n={n_stations})")
    # Shaded IQR
    ax.fill_between(feature_values, mean_curve - std, mean_curve + std, color="darkorange", alpha=0.2,label='± Std Dev')

    # Labels and grid
    ax.set_xlabel(f"{feature_of_interest} Value", fontsize=14)
    ax.set_ylabel("Predicted Water Temperature (°C)", fontsize=14)
    ax.grid(True)

    # Legend outside right
    ax.legend()

    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    return feature_values, mean_curve, std


def plot_feature_sensitivity_all_stations_aligned(
    model,
    final_df,
    station_names,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    feature_groups=None,
    title=None,
    figsize=(10, 6)
):
    """
    Plots all station sensitivity curves, aligned on the left,
    so you see relative change instead of absolute temperature.
    """
    fig, ax = plt.subplots(figsize=figsize)

    results = {}

    for station in station_names:
        res = feature_sensitivity_for_days(
            model=model,
            final_df=final_df,
            station_name=station,
            target_days=target_days,
            feature_of_interest=feature_of_interest,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            vmin=vmin,
            vmax=vmax,
            num_points=num_points,
            feature_groups=feature_groups,
            ax=None,
            plot=False
        )

        if res is None:
            print(f"Skipping {station}: no data")
            continue

        feature_values, mean_predictions, _, _ = res
        # Align on the left
        mean_predictions_aligned = mean_predictions - mean_predictions[0]
        results[station] = mean_predictions_aligned

        ax.plot(feature_values, mean_predictions_aligned, label=f"{station}")

    ax.set_xlabel(f"{feature_of_interest} Value", fontsize=14)
    ax.set_ylabel("Change in Predicted Temperature (°C)", fontsize=14)
    ax.grid(True)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=10,
        ncols=3
    )

    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    return feature_values, results


def plot_feature_sensitivity_mean_across_stations_aligned(
    model,
    final_df,
    station_names,
    target_days,
    feature_of_interest,
    x_scaler,
    y_scaler,
    vmin,
    vmax,
    num_points=50,
    feature_groups=None,
    title=None,
    figsize=(10, 6),
    save_fig=False
):
    """
    Shows the mean sensitivity curve across all stations,
    aligned on the left (predictions relative to the first feature value),
    with shaded interquartile range (25-75%).
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_curves = []
    feature_values = None
    n_stations = len(station_names)

    for station in station_names:
        res = feature_sensitivity_for_days(
            model=model,
            final_df=final_df,
            station_name=station,
            target_days=target_days,
            feature_of_interest=feature_of_interest,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            vmin=vmin,
            vmax=vmax,
            num_points=num_points,
            feature_groups=feature_groups,
            ax=None,
            plot=False
        )

        if res is None:
            print(f"Skipping {station}: no data")
            continue

        feature_values, mean_predictions, _, _ = res
        # Align curve on the left
        mean_predictions_aligned = mean_predictions - mean_predictions[0]
        all_curves.append(mean_predictions_aligned)

    if not all_curves:
        print("No data for any station.")
        return None

    all_curves = np.array(all_curves)  # shape = (n_stations, num_points)

    mean_curve = np.nanmean(all_curves, axis=0)
    std = np.nanstd(all_curves, axis=0)
    # q25 = np.percentile(all_curves, 25, axis=0)
    # q75 = np.percentile(all_curves, 75, axis=0)

    # Plot mean
    ax.plot(feature_values, mean_curve, color="darkorange", linewidth=2, label=f"Mean across stations (n={n_stations})")
    # Shaded IQR
    ax.fill_between(feature_values, mean_curve - std , mean_curve + std, color="darkorange", alpha=0.2, label='± Std Dev')

    ax.set_xlabel(f"{feature_of_interest} Value", fontsize=14)
    ax.set_ylabel("Change in Predicted Temperature (°C)", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best", fontsize=12)

    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    if save_fig:
        fig.savefig(".png", dpi=300, bbox_inches="tight")

    plt.show()

    return feature_values, mean_curve, std


