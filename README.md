# Deep Learning Experiments Repository

This Git repository comprises multiple Jupyter notebooks showcasing a diverse set of deep learning experiments. Each notebook focuses on a specific task using TensorFlow and Hugging Face's DistilBERT model. Here's an overview of the contents:

## 1. TensorFlow Experiments

### a. **Image Classification with TensorFlow:**
   - This notebook employs TensorFlow for image classification tasks on various datasets, including CIFAR10, CIFAR100, Eurosat, and MNIST. Different architectures such as CNN, ResNet, and VGG16 are utilized for enhanced performance.

### b. **Variational Autoencoder with TensorFlow:**
   - Demonstrates the implementation and training of a Variational Autoencoder (VAE) using TensorFlow. The notebook illustrates the generation of new data points from the learned latent space.

### c. **LSTM Network for IMDB Reviews with Interpretability:**
   - Implements an LSTM network for sentiment analysis on IMDB reviews using TensorFlow. Additionally, interpretability techniques like LIME TextExplainer, SHAP, and DICE counterfactuals are applied to gain insights into the model's predictions.

### d. **Timeseries Regression Model:**
   - Introduces a timeseries regression model using TensorFlow. The notebook demonstrates how TensorFlow can be employed for predicting continuous values in a timeseries setting.

### e. **Building Networks from Scratch:**
   - This code file includes the construction of various neural networks from scratch. One of the networks is dedicated to training a CNN without utilizing TensorFlow or PyTorch, providing insights into the fundamentals of neural network implementation.

## 2. DistilBERT Fine-tuning

### **Fine-tuning DistilBERT on Quora Similarity Dataset:**
   - Utilizes Hugging Face's DistilBERT model for fine-tuning on the Quora Similarity Dataset. The notebook demonstrates how pre-trained transformer models can be adapted for specific tasks through fine-tuning.

## Getting Started

To run these experiments, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/vtocitu365/tensorflow_template.git
   ```

2. Open and run the Jupyter notebooks (`*.ipynb`) in your preferred environment.

## Dependencies

Ensure the required dependencies are installed. They include TensorFlow, Hugging Face's Transformers library, Matplotlib, NumPy, and any other dependencies mentioned within the notebooks.

## Contributing

Feel free to contribute by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy exploring deep learning with TensorFlow and Hugging Face!
