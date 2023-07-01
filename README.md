# Nepali Sentiment Analysis

This repository contains code for a Nepali Sentiment Analysis model. The model is trained to predict the sentiment (positive, negative, or neutral) of Nepali text.

## Dataset

The dataset used for training this model is sourced from [Shushant/NepaliSentiment](https://huggingface.co/datasets/Shushant/NepaliSentiment) on the Hugging Face Datasets Hub. The dataset consists of labelled Nepali text samples with corresponding sentiment labels.

## Model Architecture

The model architecture used in this repository is based on the BERT (Bidirectional Encoder Representations from Transformers) model, specifically the `bert-base-multilingual-cased` variant. BERT is a powerful pre-trained language model that can be fine-tuned for various natural language processing tasks, including sentiment analysis.

The model takes Nepali text as input and produces sentiment predictions as output. It uses tokenization techniques to convert the text into numerical representations and leverages the transformer-based architecture to capture contextual relationships between words in the text.

## Usage

To use the Nepali Sentiment Analysis model:

1. Install the required dependencies listed in `requirements.txt`.

2. Load the trained model using `model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)`.

3. Tokenize your Nepali text data using the same tokenization techniques used during training.

4. Feed the tokenized input to the model and obtain the predicted sentiment labels.

5. Evaluate the model's performance using suitable metrics such as accuracy, precision, recall, or F1-score.

6. To try the model practically, open `nepali-sentiment-model-tryout.ipynb` in Jupyter Notebook.

7. In the `model-tryout.ipynb` file, provide the string you want to analyze for sentiment.

8. Run the cells in `model-tryout.ipynb` to obtain the predicted sentiment label for the provided string.

9. Optionally, you can also modify the code to analyze multiple strings or perform a batch analysis.

## Repository Structure

- `model-training.ipynb`: Jupyter Notebook containing the code for training the Nepali Sentiment Analysis model.
- `model-tryout.ipnyb` : Hupyter Notebook containing the code to tryout the Model
- `inference.ipynb`: Jupyter Notebook demonstrating how to perform inference using the trained model.
- `nepali-sentiment-model-tryout.ipynb`: Jupyter Notebook for trying the model practically with a string input.
- `requirements.txt`: List of required dependencies for running the code.
- `README.md`: This readme file provides an overview of the repository.

## Acknowledgments

The dataset used in this project is sourced from [Shushant/NepaliSentiment](https://huggingface.co/datasets/Shushant/NepaliSentiment) on the Hugging Face Datasets Hub. I would like to thank the creators and contributors for providing the dataset and making it publicly available.

![image](https://github.com/Sibindra/nepali-sentiment-analysis-model/assets/59206903/4f11b042-4de2-4506-bda9-c5ef26012b3a)


## Performance and Suggestions

During the training process, the model achieved the following accuracies for different configurations:

- For batch_size=16, learning_rate=2e-5, and epochs=10, the accuracy achieved was 88%

![image](https://github.com/Sibindra/nepali-sentiment-analysis-model/assets/59206903/7fbb2573-93f4-4139-aeba-e8afb9e4e226)

![image](https://github.com/Sibindra/nepali-sentiment-analysis-model/assets/59206903/19bc55b3-4fab-4e73-a535-b5c6712f3029)


To improve the accuracy of the model, you can consider making the following changes:

- Adjust the batch size: Try experimenting with different batch sizes to see if it has any impact on the model's performance. Sometimes, a smaller or larger batch size can lead to better results.
- Fine-tune the learning rate: The learning rate determines the step size at which the model learns from the data. You can try different learning rates to find the optimal value that improves the model's accuracy.
- Increase the number of epochs: Training the model for more epochs might help it converge to a better solution. You can try increasing the number of epochs and monitoring the performance on a validation set to determine the optimal number of epochs.

It's important to note that the accuracy of the model can be influenced by various factors, including the quality and size of the training data, the complexity of the task, and the model architecture itself. Additionally, the lack of a neutral label in the dataset might affect the model's ability to accurately classify neutral sentiment.

I encourage open-source contributors to explore ways to improve the accuracy of this model and provide additional datasets that are similar in nature to [Shushant/NepaliSentiment](https://huggingface.co/datasets/Shushant/NepaliSentiment). By collaborating and sharing insights, we can collectively enhance the performance of Nepali sentiment analysis models and contribute to the field of natural language processing.

## License

This project is licensed under the [MIT License](LICENSE).
