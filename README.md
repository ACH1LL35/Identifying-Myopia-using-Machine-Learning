# 🔓 Unlocking Myopia Detection: Leveraging Deep Learning for Accurate Diagnosis

This project investigates the use of state-of-the-art deep learning models to detect **Myopia** from retinal fundus images. We evaluate the diagnostic performance of five leading convolutional neural networks (CNNs) on the **Retinal Fundus Multi-Disease Image Dataset (RFMiD)** in both frozen and fine-tuned (unfrozen) states.

---

## 📘 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Training Methodology](#training-methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [References](#references)

---

## 🌍 Project Overview

Myopia is a growing global health concern and a major cause of vision impairment. Early detection is critical to preventing irreversible damage. This study aims to build an accurate, automated classification system using deep learning to differentiate **myopic** from **non-myopic** retinal images.

Key Goals:
- Utilize transfer learning with deep CNNs
- Compare model performance (frozen vs unfrozen)
- Recommend optimal architectures for real-world deployment

---

## 📂 Dataset

We use the **Retinal Fundus Multi-Disease Image Dataset (RFMiD)**:

- **Total Images:** 3,200 (augmented to 4,758)
- **Focus Class:** Myopia
- **Split Ratio:** 60% Train, 20% Test, 20% Validation

To address **class imbalance**, we applied data augmentation:
- 🔁 Rotation, Shift, Zoom, Shearing, Flipping
- ⬆️ Augmented from 101 to 1706 myopic images

---

## 🧠 Models Evaluated

| Model           | Description |
|------------------|-------------|
| **ResNet50**      | 50-layer residual network with shortcut connections |
| **ResNet152**     | Deeper ResNet for complex patterns |
| **MobileNetV2**   | Lightweight, optimized for mobile devices |
| **DenseNet121**   | Densely connected CNN promoting feature reuse |
| **EfficientNetB7**| Compound scaled network using SE blocks |

---

## ⚙️ Training Methodology

- **Transfer Learning**: Pre-trained on ImageNet
- Training Modes:
  - **Frozen**: Only top layers trainable
  - **Unfrozen**: All layers fine-tuned
- Common Layers:
  - `GlobalAveragePooling2D`
  - Fully Connected Dense Layers
  - `BatchNormalization`
- Optimizer: Adam
- Loss: Binary Crossentropy
- Evaluation Metric: Accuracy

---

## 📊 Results

### 🔒 Frozen Model Accuracy

| Model         | Accuracy (%) |
|---------------|--------------|
| **MobileNetV2** | **96.41**     |
| ResNet152      | 93.75         |
| ResNet50       | 93.59         |
| DenseNet121    | 91.25         |
| EfficientNetB7 | ❌ 5.00       |

### 🔓 Unfrozen (Fine-tuned) Model Accuracy

| Model         | Accuracy (%) |
|---------------|--------------|
| **ResNet152**   | **95.15**     |
| ResNet50       | 94.84         |
| MobileNetV2    | 94.06         |
| DenseNet121    | 92.34         |
| EfficientNetB7 | ❌ 5.00       |

📌 **Key Insight:** Fine-tuning significantly improves accuracy, especially in ResNet-based models.

---

## ✅ Conclusion

- 🔝 **ResNet152** was the top performer after unfreezing
- 🪶 **MobileNetV2** excelled in frozen state, ideal for edge devices
- ⚠️ **EfficientNetB7** consistently underperformed
- 🎯 Augmentation was vital for model balance
- ⚖️ Architecture choice should match compute constraints and dataset scale

---

## 👥 Contributors

- **Arnab Bishakh Sarker** (ID: 21-44464-1)  
- **Zobayer Alam** (ID: 21-44487-1)  

*Department of Computer Science*  
*American International University – Bangladesh (AIUB)*

---

## 📚 References

- RFMiD Dataset – [Link](https://dx.doi.org/10.21227/s3g7-st65)
- Bismi & Na’am, 2023 – Deep learning on fundus images  
- Du et al., 2021 – Myopic maculopathy detection  
- Litjens et al., 2017 – Deep Learning in Medical Imaging  
- WHO, 2015 – Global impact of Myopia

---

> 🧠 For academic collaborations or diagnostics automation projects, feel free to reach out!
