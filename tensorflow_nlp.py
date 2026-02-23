#!/usr/bin/env python
# coding: utf-8

# # NLP

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Reshape, GlobalMaxPool1D
from sklearn.model_selection import train_test_split
import pandas as pd
from dalex import Explainer
import shap
import lime
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Load the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

# Create a reverse word index
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode a review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Tokenize and preprocess the text data using Tokenizer
max_features = 500
maxlen = 100  # Reduced from 200 to ease memory pressure on M1
embed_size = 50  # Set your desired embedding size
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
train_text = [decode_review(text) for text in train_data]
test_text = [decode_review(text) for text in test_data]

tokenizer.fit_on_texts(train_text)
train_data = tokenizer.texts_to_sequences(train_text)
test_data = tokenizer.texts_to_sequences(test_text)

# Pad the sequences
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Define the LSTM model
def lstm_sequence_model(maxlen, max_features, embed_size, metrics):
    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)))  # dropout added to combat overfitting
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    return model

# Create and compile the model
model = lstm_sequence_model(maxlen, max_features, embed_size, metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Create TensorFlow datasets for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)  # Reduced from 32 to ease memory pressure on M1
val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Train the model on the GPU if available, otherwise fall back to CPU
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'
print(f"Training on: {device}")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
with tf.device(device):
    model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping])

# Generate predictions
y_pred = model.predict(test_data)


# In[2]:


from dice_ml import Dice, Model, Data
from dice_ml.utils import helpers

import pandas as pd
import dalex as dx
import numpy as np
import warnings

# Disable user warnings
warnings.simplefilter("ignore", category=UserWarning)
def val_dalex_nytesy(model, X_train_df, X_test_df, Y_train, column_names, sample_index=0):
    """
    Visualize model interpretability using the Dalex XAI library.

    Parameters:
        model: A trained TensorFlow model.
        X_train_df: DataFrame with training dataset features.
        X_test_df: DataFrame with test dataset features.
        sample_index: Index of the sample to explain.
    """
    X_test_df = X_test_df[:10]
    # Create a Dalex Model object
    #dx.config.environment.Verbose(DALEX_CONF={"show_warnings": False})
    exp = dx.Explainer(model, data=X_train_df, y=Y_train, model_type='classification')

    # Model performance plot (ROC)
    #exp.model_performance(verbose=False).plot(geom='roc')

    # Predict parts with Break Down method for a specific sample
    predict_parts = exp.predict_parts(X_test_df[sample_index], type='break_down', label=['c']+column_names+['s'])

    # Model parts plot
    model_parts = exp.model_parts(verbose=False)
    model_parts.plot(max_vars=5)

    # Get aspects
    asp = dx.Aspect(exp)
    asp_pps = dx.Aspect(exp, depend_method='pps')
    asp.get_aspects(h=0.1)
    asp_pps.get_aspects(n=5)

    # Model triplot
    mt = asp.model_triplot(random_state=42)
    mt.plot()

    # Predict triplot
    pt_def = asp.predict_triplot(X_test_df[sample_index], random_state=42)
    pt_def.plot()

    # Model parts plot for aspects created on threshold h=0.1
    mai = asp.model_parts(h=0.1, label='for aspects created on threshold h=0.1')
    mai.plot()


# Define a custom masker
class Masker:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_data):
        return self.model(input_data)

def predict_function(texts):
    # Preprocess the text data
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_sequences = pad_sequences(text_sequences, maxlen=maxlen)

    # Make predictions using the model
    predictions = model.predict(text_sequences)

    # Convert model predictions to class probabilities
    class_probs = np.column_stack((1 - predictions, predictions))

    return class_probs

def decode_review(text_sequence):
    return ' '.join([reverse_word_index.get(i, '?') for i in text_sequence])



# Create a Dataframe to hold your text reviews
df = pd.DataFrame({'text': [decode_review(test_data[i]) for i in range(len(test_data))]})

# Initialize the Dice object
model_interface = Model(model=model, backend="TF2", model_type='classifier')
# Convert test_data to strings and then create a DataFrame
test_data_str = [[str(item) for item in sequence] for sequence in test_data]

column_names = [f"col_{i}" for i in range(maxlen)]
#val_dalex_nytesy(model, X_train, X_val, Y_train, column_names, sample_index=0)
dataframe = np.column_stack((test_data, test_labels))
dataframe = pd.DataFrame(dataframe, columns=column_names+['outcome'])
data_interface = Data(dataframe=dataframe, continuous_features=column_names, outcome_name='outcome')

d = Dice(data_interface, model_interface, method="random")
# Choose the index of the data point for which you want to find a counterfactual explanation

# Identify misclassified examples
y_pred2 = (y_pred >= 0.5).astype(int).flatten()
misclassified_indices = np.unique(np.where(y_pred2 != test_labels)[0])
print(f"There are {len(misclassified_indices)} misclassified elements. We're only showing the first 5")
for data_idx in misclassified_indices[:5]:
    # Generate counterfactual explanation
    query_instance = dataframe.iloc[data_idx:data_idx + 1].drop(columns='outcome')

    label = int(test_labels[data_idx])
    # Generate the counterfactual explanation - desired class is the opposite of the true label
    # since we want to find what would flip the misclassified prediction
    counterfactuals = d.generate_counterfactuals(query_instance, total_CFs=1, desired_class=1 - label)
    # Print the counterfactual
    counterfactual = counterfactuals.visualize_as_dataframe(show_only_changes=True)
# Create a TextExplainer
lime_explainer = LimeTextExplainer()
# Define a prediction function


# Explain the model's prediction for test_data[0]
explanation = lime_explainer.explain_instance(test_text[0], predict_function, num_features=10, top_labels=1)

# Visualize the explanation as a matplotlib figure (Python script equivalent of show_in_notebook)
fig = explanation.as_pyplot_figure(label=explanation.available_labels()[0])
fig.suptitle(f"LIME Explanation\n{test_text[0][:120]}...", fontsize=8, wrap=True)
fig.tight_layout()
fig.savefig("lime_explanation.png", dpi=150, bbox_inches='tight')
print("LIME explanation saved to lime_explanation.png")


# In[ ]: