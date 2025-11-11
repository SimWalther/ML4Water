import json
import re
import numpy as np


class FeatureGroups:
    def __init__(self, config_path):
        self.groups, self.groups_types = self.load(config_path)
        self.columns = self._columns()
        self.columns_indices = self._columns_indices()
        self.groups_columns_indices = self._groups_columns_indices()

    def _contains_kw(self, col, kw, all_features):
        # If kw is an exact feature name, require exact match
        if kw in all_features:
            return col == kw
        # Otherwise, regex match (kw as token in string)
        return re.search(rf"(?:^|_){re.escape(kw)}(?:_|$)", col) is not None

    def _extract_first_number(self, s):
        """Extract the first occurring number from a string. Return a large value if none found."""
        match = re.search(r"\d+", s)
        return int(match.group()) if match else float("inf")

    def _columns(self):
        return np.concatenate(list(self.groups.values()), axis=None)

    def _columns_indices(self):
        return {col: idx for idx, col in enumerate(self.columns)}

    def _groups_columns_indices(self):
        column_indices = self.columns_indices

        return {
            group: [column_indices[feature] for feature in self.groups[group]]
            for group in self.groups.keys()
        }

    # def compute_X_dict(,)
        
    #     return X_dict

    def load(self, config_path):
        try:
            with open(config_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")

        feature_list = data["features"]

        groups_types = {
            group_name: rules["type"] for group_name, rules in data["groups"].items()
        }

        groups_types["input_other"] = "other"

        groups = {}
        for group_name, rules in data["groups"].items():
            necessary_keywords = rules["necessary"]
            unwanted_keywords = rules["unwanted"]

            matched_cols_temp = [
                i
                for i in feature_list
                if any(
                    self._contains_kw(i, kw, feature_list) for kw in necessary_keywords
                )
            ]

            matched_cols = [
                i
                for i in matched_cols_temp
                if not any(
                    self._contains_kw(i, kw, feature_list) for kw in unwanted_keywords
                )
            ]

            groups[group_name] = matched_cols

        # Collect remaining columns not in any group
        grouped_columns = set(col for cols in groups.values() for col in cols)
        groups["input_other"] = [
            col for col in feature_list if col not in grouped_columns
        ]

        # Sort features names by the first number that appears in the name
        groups = {
            group: sorted(groups[group], key=self._extract_first_number)
            for group in groups.keys()
        }

        return groups, groups_types

    def to_input_lists(self, train_features, test_features):
        train_inputs = [
            train_features[:, indices]
            for group, indices in self.groups_columns_indices.items()
        ]

        test_inputs = [
            test_features[:, indices]
            for group, indices in self.groups_columns_indices.items()
        ]

        return train_inputs, test_inputs
