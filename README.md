# Multilayerperception-
# Sentiment Analysis on Product Reviews using Multi-Layer Perceptron (MLP)

## Overview
This project implements a **Multilayer Perceptron (MLP)** neural network for **sentiment analysis** on product reviews. The model is trained to classify customer reviews as **positive, negative, or neutral**, helping businesses understand customer feedback more effectively.

## Dataset
The dataset consists of product reviews collected from e-commerce platforms. Each review includes:
- **Review Text**: The actual customer feedback.
- **Sentiment Label**: Categorized as Positive (1), Negative (0), or Neutral (2).

### Preprocessing Steps:
1. Tokenization and text cleaning (removal of special characters, stopwords, etc.).
2. Word embedding using **TF-IDF** or **Word2Vec**.
3. Converting text data into numerical format for MLP input.

## Model Architecture
The MLP model consists of the following layers:
- **Input Layer**: Takes preprocessed textual data as input.
- **Hidden Layers**: Fully connected layers with activation functions (ReLU, Sigmoid, or Tanh).
- **Output Layer**: A softmax layer for multi-class classification (positive, negative, neutral).

### Hyperparameters:
- **Number of Hidden Layers**: 2-3 (adjustable)
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32 or 64
- **Epochs**: 20-50 (adjustable based on performance)

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow keras nltk
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-repo/mlp-sentiment-analysis.git
cd mlp-sentiment-analysis
```
2. Run the preprocessing script:
```bash
python preprocess.py
```
3. Train the MLP model:
```bash
python train.py
```
4. Evaluate the model:
```bash
python evaluate.py
```
5. Make predictions on new reviews:
```bash
python predict.py "The product quality is excellent!"
```

## Results
- The model achieves an accuracy of **XX%** on the test set.
- Example predictions:
  - *"I love this product!" → Positive*
  - *"The quality is terrible." → Negative*

## Future Improvements
- Implementing **LSTM or Transformer models** for improved accuracy.
- Fine-tuning hyperparameters using **Grid Search or Bayesian Optimization**.
- Expanding the dataset with more diverse product categories.

