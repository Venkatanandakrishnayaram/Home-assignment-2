**Deep Learning with CNNs**
**Implementation and Explanation**

**Student Information**

Name: [Venkata Nanda Krishna Yaram]

Student ID: [700765514]

Course: [CRN23848 Nueral Networks and Deep learning]
**Project Overview**

This project covers multiple deep learning tasks using TensorFlow/Keras, NumPy, and OpenCV. We explore CNN operations, edge detection, pooling, AlexNet, and ResNet architectures through practical implementations.

**Table of Contents**

Cloud Computing for Deep Learning

Convolution with Different Parameters

CNN Feature Extraction (Edge Detection & Pooling)

AlexNet Implementation

ResNet-like Model Implementation

**1. Cloud Computing for Deep Learning**

Key Concepts

Elasticity: The ability of cloud resources to scale dynamically based on demand.

Scalability: The capability of a system to handle increasing workloads by upgrading resources.

Comparison of Cloud Services for Deep Learning

AWS SageMaker: Offers managed Jupyter notebooks, auto-scaling, and pre-built deep learning frameworks.

Google Vertex AI: Provides custom training models, AutoML, and seamless integration with TensorFlow.

Azure Machine Learning Studio: Supports automated ML, drag-and-drop training, and deep learning model deployment.

**2. Convolution with Different Parameters**

Steps Implemented:

Defined a 5×5 input matrix representing an image.

Defined a 3×3 convolutional kernel (filter).

Performed four convolution operations:

Stride = 1, Padding = 'VALID'

Stride = 1, Padding = 'SAME'

Stride = 2, Padding = 'VALID'

Stride = 2, Padding = 'SAME'

Printed the output feature maps for each case.

**3. CNN Feature Extraction (Edge Detection & Pooling)**

**Task 1: Edge Detection using Sobel Filter**

Loaded a grayscale image using OpenCV.

Applied the Sobel filter in both X and Y directions.

Displayed three images:

Original Image

Edge Detection using Sobel-X

Edge Detection using Sobel-Y

**Task 2: Max Pooling and Average Pooling**

Created a random 4×4 matrix as input.

Applied 2×2 Max Pooling to extract dominant features.

Applied 2×2 Average Pooling for feature smoothing.

Printed the original, max-pooled, and average-pooled matrices.

**4. AlexNet Implementation**

Steps Implemented:

Created a Sequential model in TensorFlow.

Added the following layers:

Conv2D (96 filters, 11×11, stride=4, ReLU)

MaxPooling (3×3, stride=2)

Conv2D (256 filters, 5×5, ReLU, padding='same')

MaxPooling (3×3, stride=2)

Conv2D (384 filters, 3×3, ReLU, padding='same')

Conv2D (384 filters, 3×3, ReLU, padding='same')

Conv2D (256 filters, 3×3, ReLU, padding='same')

MaxPooling (3×3, stride=2)

Flatten Layer

Dense (4096 neurons, ReLU, Dropout 50%)

Dense (4096 neurons, ReLU, Dropout 50%)

Output Layer (10 neurons, Softmax activation)

Printed the model summary.

**5. ResNet-like Model Implementation**

**Task 1: Residual Block Implementation**

Defined a function residual_block(input_tensor, filters):

Applied two Conv2D layers (64 filters, 3×3, ReLU).

Added a skip connection before activation.

**Task 2: ResNet-like Model**

Created a model with:

Initial Conv2D (64 filters, 7×7, stride=2, ReLU)

Two residual blocks with skip connections.

Flatten layer and a Dense layer (128 neurons, ReLU)

Output Layer (Softmax activation, 10 classes)

Printed the model summary.

**Conclusion**

Implemented fundamental CNN operations including convolution, edge detection, pooling, and deep architectures.

Compared cloud-based deep learning platforms (AWS, Google Vertex AI, Azure).

Built AlexNet and a simplified ResNet model using TensorFlow/Keras.

This project demonstrates the power of CNNs in feature extraction, classification, and advanced architectures. 
