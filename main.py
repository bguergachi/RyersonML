from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

# Data sets
DIABETES_TRAINING = "C://users/ONTARIO/PycharmProjects/DATA/DIABETES_training.csv"
DIABETES_TEST = "C://users/ONTARIO/PycharmProjects/DATA/DIABETES_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header (filename=DIABETES_TRAINING,
                                                       target_dtype=np.float32, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DIABETES_TEST,
                                                   target_dtype=np.float32, features_dtype=np.float32)

#feature columns
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/DIABETES_model",feature_columns=feature_columns)

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=10000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


# Classify two new DIABETES tumor samples.
'''
new_samples = np.array(
    [], dtype=float)
y = classifier.predict(new_samples)
for i in y:
    print (i)
'''
