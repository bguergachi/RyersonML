import tensorflow as tf
import numpy as np
import pandas as pd

# Data sets
TF_DATA_FILE = "C://users/ONTARIO/Desktop/TFData.csv"
DIABETES_TRAINING = "C://users/ONTARIO/Desktop/TFData_training.csv"
DIABETES_TEST = "C://users/ONTARIO/Desktop/TFData_test.csv"

#Creating dataset
dset = pd.read_csv(TF_DATA_FILE,sep=",",usecols=('sBP','LDL','HDL','FBS','Diabetes'))
print ("datasize",dset.size)
dset.replace('', np.nan, inplace = True)
dset.dropna(inplace=True)
dset.to_csv(DIABETES_TRAINING,sep=',',index=False,header=False)

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header (filename=DIABETES_TRAINING,
                                                       target_dtype=np.int, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DIABETES_TEST,
                                                   target_dtype=np.int, features_dtype=np.float32)

#feature columns
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 10],
                                            n_classes=2,
                                            model_dir="diabetes_model",feature_columns=feature_columns)
print('Training Neural Network....')
# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=100000)
print('Training completed!')

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


# Classify two new patient samples.
new_samples = np.array(
    [[121,3.8,1.36,5.4,0.63,5,5.45,1]], dtype=float)
y = classifier.predict(new_samples,as_iterable=False)

print('Inputs: {}'.format(new_samples))
print ('Predictions: {}'.format(y))

