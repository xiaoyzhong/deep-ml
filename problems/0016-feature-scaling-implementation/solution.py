import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Calculate mean and standard deviation for each feature
	mean=np.mean(data, axis=0)
	std=np.std(data, axis=0)

    # Standardization
	standardized_data = (data - mean)/std

    # Calculate min and max for each feature
	min_values = np.min(data, axis=0)
	max_values = np.max(data, axis=0)

    # Min-max normalization
	normalized_data = (data - min_values) / (max_values - min_values)

    # Round results to 4 decimal places
	standardized_data = np.round(standardized_data, 4)
	normalized_data = np.round(normalized_data, 4)

	return standardized_data, normalized_data