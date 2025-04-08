"""
Modul ini berisi fungsi untuk melakukan preprocessing dataset menggunakan TFX Transform.
"""

import tensorflow as tf
import tensorflow_transform as tft

# Daftar numerical fitur pada dataset
NUMERICAL_FEATURES = [
    "TotalWorkingYears",
    "Age",
    "MonthlyIncome",
]

# Daftar categorical fitur pada dataset
CATEGORICAL_FEATURES = {
    "OverTime": 2,  # Yes/No -> Binary (2 unique values)
    "MaritalStatus": 3,  # Married/Single/Divorced -> 3 unique values
    "JobRole": 9,  # Healthcare Representative, Sales Executive, etc. -> 9 unique values
    "Department": 3,  # Research & Development, Sales, etc. -> 3 unique values
}

# Label key
LABEL_KEY = "Attrition"

# Fungsi untuk mengubah nama fitur yang sudah ditransformasi
def transformed_name(key):
    """Renaming transformed features."""
    return key + "_xf"

# Fungsi untuk melakukan preprocessing
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: Map dari kunci fitur ke raw features.

    Returns:
        outputs: Map dari kunci fitur ke fitur yang telah ditransformasi.
    """
    outputs = {}

    # Transformasi untuk fitur numerik
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    # Transformasi untuk fitur kategorikal
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature]
        )

    # Transformasi untuk label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tft.compute_and_apply_vocabulary(inputs[LABEL_KEY]), tf.int64
    )

    return outputs
