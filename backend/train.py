# Code from https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html
# --------------------------------------
# Initialization
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda, Embedding
from tensorflow.keras.models import Model
from tqdm import tqdm
import pandas as pd
import shap

# print the JS visualization code to the notebook
shap.initjs()


# --------------------------------------
# Load dataset
data = pd.read_csv('data/dataset.csv')   # dataset location
i = 8                                                       # index of label

# Select labels
X = data.drop(data.columns[i],axis=1)
y = data.iloc[:, 8]

# X,y = shap.datasets.adult()
# normalize data (this is important for model convergence)
dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
for k,dtype in dtypes:
    if dtype == 'float32':
        X[k] -= X[k].mean()
        X[k] /= X[k].std()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)


# --------------------------------------
# Train Keras model

# build model
input_els = []
encoded_els = []
for k,dtype in dtypes:
    input_els.append(Input(shape=(1,)))
    if dtype == 'int8':
        e = Flatten()(Embedding(X_train[k].max()+1, 1)(input_els[-1]))
    else:
        e = input_els[-1]
    encoded_els.append(e)
encoded_els = concatenate(encoded_els)
layer1 = Dropout(0.5)(Dense(100, activation='relu')(encoded_els))
out = Dense(1)(layer1)

# train model
regression = Model(inputs=input_els, outputs=[out])
regression.compile(optimizer='adam', loss='binary_crossentropy')
regression.fit(
    [X_train[k].values for k,t in dtypes],
    y_train,
    epochs=50,
    batch_size=512,
    shuffle=True,
    validation_data=([X_valid[k].values for k,t in dtypes], y_valid)
)

regression.save('/model/trained-regressor')
