# Text_Classification_Using_BERT_
Spam email filtering using BERT

## Required libraries
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text
    from sklearn.model_selection import train_test_split

## Data Preprocess
The dataset is downloaded from Kaggle as a CSV file - spam.csv

## BERT model
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
Using these URLs, the preprocess and encoder models are implemented.

## Sentence embedding
    def get_sentence_embedding(sentence):
      preprocess_model = hub.KerasLayer(preprocess_url)
      preprocessed_text = preprocess_model(sentence)
  
      bert_model = hub.KerasLayer(encoder_url)
      outputs = bert_model(preprocessed_text)
      return outputs["pooled_output"]

The BERT model gives four outputs. ['sequence_output', 'pooled_output', 'default', 'encoder_outputs']\
      ===> pooled_output   : sentence embedding \
      ===> sequence_output : word embedding

## Building a functional model
     # BERT layers
        text_input = tf.keras.layers.Input(shape=(), dtype = tf.string, name = "text")
        preprocessed_text = preprocess_model(text_input)
        outputs = bert_model(preprocessed_text)
      
      # Neural network layers
        input_layer = tf.keras.layers.Dropout(0.1, name = "dropout")(outputs["pooled_output"])
        output_layer = tf.keras.layers.Dense(1, activation = "sigmoid", name ="output")(input_layer)
      
      # construct final model
      
        model = tf.keras.Model(inputs = [text_input], outputs = [output_layer])

## Training
      loss: 0.2892 - accuracy: 0.9062 - precision: 0.8916 - recall: 0.9250

## Testing
      loss: 0.2723 - accuracy: 0.9091 - precision: 0.8964 - recall: 0.9251

## Confusion matrix

![image](https://github.com/priyanthan07/Text_Classification_Using_BERT_/assets/129021635/2831d59d-5d10-48f7-94ef-33f5887e68a7)

