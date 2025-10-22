# logRegressionSpeedupParallelization

## Project Title & Description

This repository contains a Python implementation of L2 regularized Logistic Regression, focusing on speedup through parallelization techniques, including CUDA acceleration. The project explores different optimization strategies and their impact on performance.

## Key Features & Benefits

*   **L2 Regularized Logistic Regression:** Implementation of Logistic Regression with L2 regularization to prevent overfitting.
*   **Parallelization with CUDA:** Leveraging CUDA for accelerated computation, particularly in gradient calculations and loss function evaluation.
*   **Cross-Validation:** Includes functionality for hyperparameter tuning using cross-validation.
*   **Dataset Loading:** Utilities for loading various datasets in different formats.
*   **Performance Analysis:** Designed to compare the performance of different optimization techniques.

## Prerequisites & Dependencies

Before running the code, ensure you have the following installed:

*   **Python 3.6+**
*   **NumPy:** For numerical computations.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn (sklearn):** For machine learning algorithms and utilities.
*   **Numba:** For CUDA acceleration.
*   **CUDA Toolkit:** For CUDA-enabled GPU usage (if using CUDA features).

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn numba
```

If you intend to use the CUDA accelerated version, ensure you have the CUDA Toolkit installed and configured correctly.  Refer to the NVIDIA documentation for installation instructions.

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MarcoTrambusti/logRegressionSpeedupParallelization.git
    cd logRegressionSpeedupParallelization
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies (if you didn't do it earlier):**

    ```bash
    pip install -r requirements.txt # If requirements.txt exists (if not see Dependencies above)
    ```

4.  **Configure datasets path (if necessary):**
    The `config.py` file specifies the location of the datasets.  Ensure that the `DATASETS_FOLDER` variable points to the correct directory. The default path is 'Datasets'.

## Usage Examples

### Cross-Validation

Run cross-validation to find the best lambda value:

### Running Logistic Regression with CUDA


## Configuration Options

The `config.py` file contains the following configuration options:

*   `DATASETS_FOLDER`: The path to the directory containing the datasets.  Default: `'Datasets'`
*   `CV_RESULTS_PATH`: The path to save the cross-validation results. Default: `'cv_results.csv'`
*   `RESULT_FOLDER`: The path to the result folder. Default: `'results'`
*   `TEST_RESULTS_PATH`: The path to the test result file. Default: `'results.csv'`

You can modify these variables directly in the `config.py` file.


*   The datasets used in this project are publicly available and were obtained from various sources. See the individual dataset descriptions for more information.
*   The implementation is inspired by various online resources and tutorials on Logistic Regression and CUDA programming.
