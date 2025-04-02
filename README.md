# Face-Mask-Detection-using-CNN

This project uses Convolutional Neural Networks (CNNs) to classify images into two categories: **With Mask** and **Without Mask**. The model was trained on a dataset containing over 12,000 images of people wearing face masks and not wearing face masks.

## Dataset

The dataset used in this project is the [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) from Kaggle. You need to download it manually from Kaggle before using it. The dataset is divided into three folders:

- **train**: Contains the training images.
- **validation**: Contains images for validating the model during training.
- **test**: Contains images for testing the model after training.

The dataset contains two classes:

- **with_mask**
- **without_mask**

To download the dataset, you need to have a Kaggle account. Once logged in, follow these steps:

1. Go to the [dataset page](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset).
2. Click on the "Download" button to download the zip file.
3. Extract the contents into a folder named **`Face Mask Dataset`** with the following subfolders:
   - `train`
   - `validation`
   - `test`

## Requirements

Make sure you have the following libraries installed:

- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pillow

You can install the required dependencies using the following command:

```bash
pip install tensorflow matplotlib numpy pillow
```

## Model Architecture

The model is built using Convolutional Neural Networks (CNN). The architecture includes:

- Convolutional layers (Conv2D) to extract features from images.
- MaxPooling layers to downsample the feature maps.
- Fully connected layers (Dense) to make predictions.
- Dropout for regularization to prevent overfitting.

The model is compiled with the Adam optimizer and the binary cross-entropy loss function. It was trained for 10 epochs using data augmentation to improve generalization.

## Training

The model is trained on the dataset using the following configuration:

- Image dimensions: 150x150
- Batch size: 32
- Epochs: 10
- Data Augmentation: Applied to the training set for rotation, shifts, zoom, and flips.

## Model Performance

After training, the model achieved the following results:

- **Final Training Accuracy**: 93.75%
- **Final Validation Accuracy**: 97.12%
- **Test Accuracy**: 96.17%

## Usage

Once the model is trained, it can be used for real-time face mask detection. You can use it for applications such as:

- Monitoring compliance in public places.
- Building face mask detection systems for healthcare settings.
- Developing real-time applications using mobile or web frameworks.

## Conclusion

This project demonstrates how to use CNNs for face mask detection. The model achieves high accuracy and can be deployed in various real-world applications to help ensure public safety.

## License

This project is licensed under the MIT License.
