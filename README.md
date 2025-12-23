# fish image classification

## 1️⃣ Dataset Upload (Kaggle → Colab)

* First, the dataset was available as a ZIP file.
* The ZIP file was uploaded to Kaggle.
* From Kaggle, the dataset was loaded into Google Colab using the Kaggle API.
* After loading, the dataset was unzipped to access image folders.

*Now the images are ready for CNN training.*

## 2️⃣ CNN Model Architecture (Custom CNN)

* This project uses a Custom Convolutional Neural Network (CNN).
* The model is built using the Keras Sequential API.
* **Input image size:** 224 × 224 × 3 (RGB images)
* **Total number of output classes:** 11

## 🧠 Model Architecture Breakdown (Custom CNN)

### 🔹 Sequential Model
- Layers are stacked **one after another** in a sequential manner.

---

### 🔹 Block 1
**Components:**
- 32 convolutional filters (Conv2D)
- ReLU activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (30%)

**Explanation:**
- Detects basic features like edges and corners
- ReLU removes negative values
- BatchNormalization stabilizes and speeds up training
- MaxPooling reduces image size
- Dropout prevents overfitting

---

### 🔹 Block 2
**Components:**
- 64 convolutional filters (Conv2D)
- ReLU activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (30%)

**Explanation:**
- Increased filters (64) learn more detailed patterns
- Normalization, pooling, and dropout provide stability and regularization

---

### 🔹 Block 3
**Components:**
- 128 convolutional filters (Conv2D)
- ReLU activation
- Batch Normalization
- MaxPooling (2×2)
- Dropout (40%)

**Explanation:**
- 128 filters learn complex features
- Higher dropout (40%) reduces overfitting

---

### 🔹 Fully Connected Layer
**Components:**
- Flatten layer
- Batch Normalization
- Dense layer with 128 neurons
- Dropout (50%)

**Explanation:**
- Flatten converts feature maps into a 1D vector
- Dense layer learns high-level representations
- Dropout provides strong regularization

---

### 🔹 Output Layer
**Components:**
- Dense layer with 11 neurons
- Softmax activation

**Explanation:**
- 11 neurons represent 11 classes
- Softmax outputs probabilities for each class
- The class with the highest probability is the final prediction


## ⚙️ Model Compilation

- The model was compiled using the **Adam optimizer**, which adapts the learning rate during training for better convergence.
- **Categorical Crossentropy** was used as the loss function because this is a **multi-class classification problem**.
- **Accuracy** was chosen as the evaluation metric to track how well the model classifies fish images.

---

## 🏋️ Model Training

- The model was trained for **100 epochs**, allowing it to learn features from the training dataset effectively.
- During training, the model gradually adjusted its weights to minimize the loss and improve accuracy.
- Validation data was used to monitor performance and prevent overfitting.

---

## 🧪 Model Testing and Evaluation

- After training, the model was evaluated on a **test dataset** that it had never seen before.
- The evaluation measured both the loss and the accuracy on this unseen data.
- The model achieved **96.90% accuracy**, indicating that it generalized very well to new fish images.

---

## 📊 Model Performance

- **High accuracy (96.90%)** shows the model is effective at distinguishing between the 11 fish categories.
- The CNN successfully learned complex features such as shapes, colors, and patterns of different fish species.
- The performance suggests the model is reliable for **multi-class fish image classification** tasks.

---

## ✅ Summary

- Model compilation, training, and testing were completed successfully.
- Achieved **excellent performance** on unseen test data.
- Ready for further improvements, optimization, or deployment in real-world applications.

## 🐟 Fish Image Classification using Pretrained VGG16

### 🔹 Using Pretrained Model

- Instead of building a CNN from scratch, a **pretrained VGG16 model** was used.
- VGG16 is a well-known **deep CNN** trained on **ImageNet** with millions of images.
- The **base layers** of VGG16 were used for feature extraction.
- The top (classification) layers of VGG16 were removed using `include_top=False`.
- **Weights were frozen** so that only the newly added layers are trainable, preventing the base from being modified.

---

### 🔹 Adding Custom Layers

- After the base model, a **Global Average Pooling layer** was added to reduce the spatial dimensions of the feature maps.
- A **Dense layer with 256 neurons and ReLU activation** was added to learn high-level representations.
- **Dropout (50%)** was added for regularization to prevent overfitting.
- A **final Dense layer with 11 neurons and softmax activation** was added to classify the 11 fish categories.

---

### 🔹 Model Compilation

- The model was compiled using the **Adam optimizer**.
- **Categorical Crossentropy** was used as the loss function because it is a **multi-class classification problem**.
- **Accuracy** was used as the evaluation metric to measure model performance.

---

### 🔹 Model Training

- The model was trained for **25 epochs** using the prepared training and validation datasets.
- The pretrained VGG16 base helped the model **learn faster and more accurately**, as it already contains rich image features.
- Validation data was used to monitor performance and prevent overfitting.

---

### 🔹 Model Performance

- **Training Accuracy:** 95.17%  
- **Test Accuracy:** 97.77%  

**Observations:**
- The model generalized very well to unseen fish images.
- Pretrained VGG16 features significantly improved performance compared to the custom CNN.
- High test accuracy indicates strong capability for multi-class fish classification.

---

### ✅ Summary

- Pretrained VGG16 was used for feature extraction.
- Custom Dense and Dropout layers were added for classification.
- Achieved **excellent accuracy (97.77%)** on the test dataset.
- Ready for deployment or further optimization.

## 🐟 Fish Image Classification using Pretrained MobileNetV2

### 🔹 Using Pretrained Model

- A **pretrained MobileNetV2** model was used instead of building a CNN from scratch.
- MobileNetV2 is a lightweight deep CNN trained on **ImageNet**, suitable for faster training and efficient inference.
- The **base layers** were used for feature extraction.
- The top classification layers were removed (`include_top=False`) to add custom layers.
- **Weights of the base model were frozen**, so only newly added layers were trainable.

---

### 🔹 Adding Custom Layers

- A **Global Average Pooling layer** was added after the base model to reduce the feature maps.
- A **Dense layer with 256 neurons and ReLU activation** was added for learning high-level representations.
- **Dropout (50%)** was added to prevent overfitting.
- A **final Dense layer with 11 neurons and softmax activation** was added for multi-class fish classification.

---

### 🔹 Model Compilation

- Compiled using the **Adam optimizer**.
- Loss function: **Categorical Crossentropy** (for multi-class classification).
- Metric: **Accuracy** to evaluate performance.

---

### 🔹 Model Training

- Trained for **25 epochs** on the training and validation datasets.
- MobileNetV2’s pretrained weights helped the model **learn faster** and achieve **high accuracy**.
- Validation data was used to monitor performance and avoid overfitting.

---

### 🔹 Model Performance

- **Training Accuracy:** 98.81%  
- **Test Accuracy:** 99.43%  

**Observations:**
- The model generalized extremely well to unseen fish images.
- MobileNetV2’s pretrained features significantly improved both **accuracy** and **training speed**.
- High test accuracy indicates excellent capability for multi-class fish classification.

---

### ✅ Summary

- MobileNetV2 pretrained model used for feature extraction.
- Custom Dense and Dropout layers added for final classification.
- Achieved **near-perfect test accuracy (99.43%)**.
- Ready for deployment or further optimization.

## 🐟 Fish Image Classification using Pretrained DenseNet121

### 🔹 Using Pretrained Model

- A **pretrained DenseNet121** model was used for fish image classification.
- DenseNet121 is a deep CNN trained on **ImageNet**, known for its **dense connections** that improve feature reuse and gradient flow.
- The **base layers** were used for feature extraction.
- The top classification layers were removed (`include_top=False`) to add custom layers.
- **Base model weights were frozen**, so only the added layers were trainable.

---

### 🔹 Adding Custom Layers

- **Global Average Pooling layer** added to reduce the spatial dimensions of feature maps.
- **Dense layer with 256 neurons and ReLU activation** added for high-level representation learning.
- **Dropout (50%)** added to reduce overfitting.
- **Final Dense layer with 11 neurons and softmax activation** added for multi-class classification of fish species.

---

### 🔹 Model Compilation

- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy** (for multi-class classification)
- Metric: **Accuracy** to evaluate performance

---

### 🔹 Model Training

- Trained for **25 epochs** using training and validation datasets.
- Pretrained DenseNet121 features helped the model **learn complex fish features efficiently**.
- Validation data monitored during training to prevent overfitting.

---

### 🔹 Model Performance

- **Training Accuracy:** 98.85%  
- **Test Accuracy:** 99.71%  

**Observations:**
- The model generalized extremely well to unseen fish images.
- DenseNet121 achieved the **highest accuracy among all models** tested.
- Excellent feature extraction and strong generalization make it ideal for multi-class fish classification.

---

### ✅ Summary

- DenseNet121 pretrained model used for feature extraction.
- Custom Dense and Dropout layers added for classification.
- Achieved **best test accuracy (99.71%)** among all models (Custom CNN, VGG16, MobileNetV2, DenseNet121).
- Selected as the **best-performing model** for deployment and further optimization.

## 🐟 Fish Image Classification using Pretrained ResNet50

### 🔹 Using Pretrained Model

- A **pretrained ResNet50** model was used for fish image classification.
- ResNet50 is a deep CNN trained on **ImageNet**, known for its **residual connections** that help with very deep networks.
- The base layers were used for **feature extraction**.
- Top classification layers were removed (`include_top=False`) to add custom layers.
- **Initially, all base model weights were frozen**, so only newly added layers were trainable.

---

### 🔹 Adding Custom Layers

- A **Global Average Pooling layer** was added after the base model to reduce feature map dimensions.
- A **Dense layer with 256 neurons and ReLU activation** was added to learn high-level features.
- **Dropout (50%)** added to reduce overfitting.
- **Final Dense layer with 11 neurons and softmax activation** added for multi-class fish classification.

---

### 🔹 Initial Model Training

- The model was trained for **20 epochs** with frozen base layers.
- **Initial Training Performance:**
  - Training Accuracy: 31.43%
  - Test Accuracy: 41.60%
- Observations:
  - The model struggled to learn effectively when the base layers were frozen.
  - ResNet50 requires **fine-tuning** for this dataset to perform better.

---

### 🔹 Fine-Tuning ResNet50

- Fine-tuning was applied by keeping the base weights frozen initially and training the added layers.
- After fine-tuning, performance improved slightly.

- **Fine-Tuned Model Performance:**
  - Training Accuracy: 31.53%
  - Test Accuracy: 39.15%
- Observations:
  - Fine-tuning improved performance only marginally.
  - ResNet50 performed significantly worse compared to DenseNet121, MobileNetV2, and VGG16 on this fish dataset.

---

### ✅ Summary

- Pretrained ResNet50 used for feature extraction and classification.
- Initial training with frozen base layers showed poor performance.
- Fine-tuning improved performance slightly, but accuracy remained low.
- ResNet50 is **not suitable for this multi-class fish image classification** compared to other models tested.

## 🐟 Fish Image Classification using Pretrained EfficientNetB0

### 🔹 Using Pretrained Model

- A **pretrained EfficientNetB0** model was used for fish image classification.
- EfficientNetB0 is a modern CNN known for **efficient scaling of depth, width, and resolution**.
- Base layers were used for feature extraction with pretrained ImageNet weights.
- Initially, the **top layers were frozen**, and custom Dense + Dropout layers were added.

---

### 🔹 Model Performance (Before Fine-Tuning)

- **Training Accuracy:** 16.33%  
- **Test Accuracy:** 16.31%  

**Observations:**
- The model **performed very poorly** compared to VGG16, MobileNetV2, and DenseNet121.
- Likely reasons:
  - Fish dataset size may be too small for EfficientNetB0’s architecture.
  - Base layers frozen → pretrained features not suitable for this dataset.
- Base pretrained features did not generalize well to fish images.

---

### 🔹 Fine-Tuning EfficientNetB0

- The model was **fine-tuned** by unfreezing some of the base layers to adapt to the fish dataset.
- Additional training was performed to update weights in selected base layers.

---

### 🔹 Model Performance (After Fine-Tuning)

- **Training Accuracy:** 17.05%  
- **Test Accuracy:** 16.31%  

**Observations:**
- Fine-tuning slightly improved training accuracy but **test accuracy remained almost the same**.
- Indicates the model **struggled to learn meaningful fish-specific features**.
- Overall, EfficientNetB0 **behaved poorly** on this dataset.

---

### ✅ Summary

- EfficientNetB0 was tested but **not suitable** for this fish classification task.
- Even fine-tuning could not improve test accuracy significantly.
- Compared to DenseNet121, MobileNetV2, and VGG16, EfficientNetB0 was **the worst performer** on this dataset.

## 💾 Saving the DenseNet121 Model

- The best-performing DenseNet121 model (test accuracy 99.71%) was **saved using the Keras native `.keras` format**.
- File name: `dense_model.keras`
- Saving the model allows:
  - **Reloading the model later** without retraining.
  - **Deployment in applications** like Streamlit.
  - Sharing the model for other uses or experiments.

### 🔹 Streamlit Interface

- A **Streamlit web app** was created for user interaction.
- Users can **upload a fish image** through the interface.
- The uploaded image is **processed and passed through the DenseNet121 model**.
- The model predicts the **fish class** (one of the 11 categories) and displays it on the screen.

---

### 🔹 Features of the Interface

- User-friendly **drag-and-drop or browse image upload**.
- Shows the **predicted class label** for the uploaded fish image.
- Uses **pretrained DenseNet121** for accurate predictions.
- Enables **real-time testing** of the model on new images.

---

### ✅ Summary

- DenseNet121 saved and deployed via Streamlit for **real-time fish classification**.
- Users can **easily test the model** with their own images.
- Provides a **practical demonstration** of multi-class fish classification.
