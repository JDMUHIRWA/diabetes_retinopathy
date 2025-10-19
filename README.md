# Diabetic Retinopathy Classification

This repository contains code, notebooks, and documentation for a comprehensive project on automated diabetic retinopathy (DR) detection using machine learning and deep learning techniques. The project compares traditional ML models (Random Forest, SVM, XGBoost) with deep learning architectures (Custom CNN, VGG16, ResNet50) on a large, real-world retinal image dataset.

## Project Overview
- **Goal:** Develop and evaluate models for early detection of diabetic retinopathy from retinal fundus images, prioritizing clinical sensitivity and practical deployment in resource-limited settings.
- **Dataset:** Kaggle Diabetic Retinopathy Detection (35,126 images, binary labels: No DR / Has DR)
- **Approaches:**
  - Traditional ML: Random Forest, SVM, XGBoost (handcrafted features)
  - Deep Learning: Custom CNN, VGG16 (transfer learning), ResNet50 (fine-tuned)
- **Key Techniques:**
  - Advanced image preprocessing and augmentation
  - Feature engineering (HOG, color histograms, texture, shape)
  - Class imbalance handling (focal loss, class weights, threshold tuning)
  - Clinical metric prioritization (recall, sensitivity)

## Repository Structure
- `notebooks/` — Main analysis notebook
- `visualisations/` — Figures and diagrams for results and report
- `README.md` — Project overview and instructions

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/JDMUHIRWA/diabetes_retinopathy.git
   cd diabetes_retinopathy
   ```
2. **Download the dataset:**
   - [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
   - Place images in the `data/` directory.
3. **Install dependencies:**
   - Python 3.8+
   - TensorFlow, Keras, scikit-learn, pandas, numpy, matplotlib, seaborn
   - Install with:
     ```bash
     pip install -r requirements.txt
     ```
4. **Run the notebook:**
   - Open `Diabetes Retinopathy.ipynb` in Jupyter or VS Code
   - Run all cells top-to-bottom

## Results Summary
- **Best Model:** Custom CNN baseline (AUC 0.617, Recall 60%, F1 43.6%)
- **Key Insights:**
  - Deep learning models outperform traditional ML in sensitivity and adaptability
  - Class imbalance handling and threshold optimization are critical for clinical deployment
  - Efficient CNN architectures can rival transfer learning models with lower computational cost

## Figures & Visualizations
- Confusion matrices, ROC curves, training curves, and feature importance plots are available in `visualisations/`.

## Academic Report
- The full report is included in the repository, detailing methodology, results, discussion, and references.



## Contact
For questions or collaboration, contact [JDMUHIRWA](j.hareriman@alustudent.com).
