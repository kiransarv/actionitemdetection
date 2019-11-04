from tensorflow import keras;
from ml.actionitem_detection import read, tranform_labels;

import sys;

tokenizer = keras.preprocessing.text.Tokenizer(lower=True, num_words=2000);

labels, docs = read("/home/kiran/Datasets/Huddl/dataset");
y = tranform_labels(labels);

tokenizer.fit_on_texts(docs);
seqs = tokenizer.texts_to_sequences(docs);
X = keras.preprocessing.sequence.pad_sequences(seqs, maxlen=50, padding="post");

vocab_size = len(tokenizer.word_index) + 1;
print(vocab_size);

es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5);

model = keras.models.Sequential();
model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=50));
model.add(keras.layers.GlobalAveragePooling1D());
#model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"));
model.add(keras.layers.Dropout(0.2));
model.add(keras.layers.Dense(1, activation="sigmoid"));
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]);
model.summary();

model.fit(X, y, epochs=100, validation_split=0.1, batch_size=100, callbacks=[es]);