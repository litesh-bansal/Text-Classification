# Text-Classification

Here’s an analysis of your code and the process of performing sentiment analysis using RNN and LSTM models:

### 1. **Data Preprocessing**
   - **Text Cleaning**: 
     You remove non-alphabetical characters using regex (`re.sub`), tokenize the text into words, and remove stopwords like "and," "the," etc., to reduce noise. 
     - **Lemmatization**: Reduces words to their base forms (e.g., "running" to "run") using `WordNetLemmatizer`. This is useful in NLP as it reduces the vocabulary size and improves model generalization.
   
   - **Tokenization and Padding**:
     - The `Tokenizer` class from Keras is used to convert text data into sequences of integers. The tokenizer learns a vocabulary of all unique tokens (words) in the dataset and assigns an integer to each word.
     - **Padding**: After tokenizing, the sequences are padded to ensure uniform input length, which is essential for feeding data into neural networks like RNNs and LSTMs that expect fixed-length input.

### 2. **Dataset Preparation**
   - **Train-Validation Split**: 
     - The data is split into training and validation sets using `train_test_split`. 
     - Training data is used to train the models, while validation data helps evaluate model performance and avoid overfitting.
   - **TensorDataset** and **DataLoader**:
     - `TensorDataset` wraps tensors (inputs and labels), and `DataLoader` allows efficient batching, shuffling, and loading of data during training, enhancing model performance.

### 3. **Model Definitions**
   - **Recurrent Neural Network (RNN)**:
     - An RNN model is defined using PyTorch’s `nn.Module`. 
     - The key layers include:
       - **Embedding Layer**: Transforms integer-encoded words into dense vectors of fixed size, which are trainable during the model’s learning process.
       - **RNN Layer**: Processes the sequence of embeddings and maintains a hidden state across time steps, allowing it to learn dependencies in the sequence.
       - **Dropout Layer**: A dropout rate of 50% is used to prevent overfitting.
       - **Fully Connected Layer**: Maps the output to the final sentiment prediction.

   - **Long Short-Term Memory (LSTM)**:
     - LSTM extends RNN by adding mechanisms to maintain long-term dependencies using forget, input, and output gates, addressing the vanishing gradient problem common in RNNs.
     - Similar to the RNN, it uses an embedding layer, dropout, and a fully connected layer to predict the sentiment.

### 4. **Model Initialization and Summary**
   - The vocabulary size is calculated based on the tokenizer's word index.
   - Both the RNN and LSTM models are initialized with:
     - Output size: 3 (for 3 sentiment classes—positive, negative, neutral).
     - Embedding dimension: 400 (the size of the word vectors).
     - Hidden dimension: 256 (the number of units in each RNN/LSTM cell).
     - Number of layers: 2 (for stacking RNN/LSTM cells).
   - Model summaries are printed, providing insights into the architecture of both models.

### 5. **Loss Function and Optimizers**
   - **Loss Function**: `nn.CrossEntropyLoss` is used as it’s suitable for multi-class classification problems (like sentiment analysis with 3 classes).
   - **Optimizer**: Adam optimizer is chosen for both models. Adam combines the advantages of both Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp), making it efficient for this task.

### 6. **Training and Validation Process**
   - The `train_and_validate` function handles the training and validation for both models.
     - **Training Loop**: 
       - The model is set to training mode, and batches of data are fed through the model.
       - For each batch, the predictions are made, the loss is calculated, and the gradients are backpropagated. The optimizer then updates the model’s parameters.
     - **Validation Loop**:
       - After training, the model is evaluated on the validation set in evaluation mode (to prevent updates to weights).
       - **Accuracy Calculation**: Accuracy is calculated by comparing the model’s predictions with the ground truth.

   - **Epochs**: 
     - You train the models for 20 epochs. Each epoch represents one complete pass through the training dataset.
     - Loss and accuracy are printed for each epoch, allowing you to track the model’s performance over time.

### 7. **Plotting Losses**
   - You visualize the training and validation loss curves using `matplotlib`. This helps identify whether the models are overfitting (if validation loss increases while training loss decreases) or underfitting (if both training and validation loss remain high).

### 8. **Training Results**:
   - Both models improve validation accuracy and decrease loss over epochs. Notably:
     - The **RNN model** achieves decent accuracy but tends to overfit slightly towards the later epochs.
     - The **LSTM model** generally outperforms RNN, maintaining better accuracy on the validation set, thanks to its ability to capture long-term dependencies.

### 9. **Building a GUI for Sentiment Prediction**
   - You implemented a basic GUI using `tkinter` to allow users to input a review and receive sentiment predictions from the trained models.
   - **Predict Function**: 
     - The `predict_sentiment` function processes the user input, tokenizes and pads the sequence, and passes it to the model for prediction.
     - The predicted sentiment class (positive, negative, or neutral) is displayed as the output.

---

### Key Takeaways:
- **LSTMs outperform simple RNNs** for text data as they better handle long-range dependencies, a crucial feature for sentiment analysis where context matters.
- **Data preprocessing** is essential for text models, as clean, lemmatized, and tokenized text improves model performance significantly.
- **GPU acceleration** using CUDA ensures that large models like LSTMs can be trained efficiently.


