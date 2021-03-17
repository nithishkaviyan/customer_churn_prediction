"""
    This script contains utils functions used in modelling
"""

# Load packages
import numpy as np


# This helper function explains the confusion matrix obtained from confusion_matrix method
def confusion_matrix_report(confusion_arr: np.ndarray):
    """
        Function to explain the confusion matrix

        Parameters
        -----------
        confusion_arr  :   np.ndarray containing values of confusion matrix

        Returns a dictionary explaining each value in confusion_arr
    """
    return {"True Positive": confusion_arr[1, 1],
            "False Positive": confusion_arr[0, 1],
            "True Negative": confusion_arr[0, 0],
            "False Negative": confusion_arr[1, 0]}
