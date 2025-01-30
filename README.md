# Anomaly Detection using Evolutionary Learning and Ensemble Scoring

## Objective
This project tackles the challenge of fraud detection using the [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection). By leveraging evolutionary algorithms through DEAP, we create and optimize a diverse population of classifiers to detect fraudulent transactions. The ultimate goal is to develop an ensemble scoring system that combines predictions from multiple classifiers, offering robust and adaptive fraud detection.

---

## Key Features & Benefits
- **Diverse Classifier Population**: Maintains a heterogeneous mix of classifiers (e.g., Random Forest, SVM, XGBoost) through mutation-based evolutionary optimization.
- **Ensemble Scoring**: Combines predictions using weighted voting or confidence aggregation, dynamically optimized via a secondary DEAP process.
- **Adaptive Optimization**: Continuously evolves hyperparameters and classifier weights to improve detection accuracy.
- **Real-World Applicability**: Designed for large-scale, production-ready fraud detection pipelines.

---

## Implementation

### Tech Stack
- **TensorFlow Extended (TFX)**: Defines and manages the ML pipeline, covering data ingestion, preprocessing, training, and evaluation.
- **Airflow**: Orchestrates the workflow and schedules pipeline tasks.
- **MySQL**: Stores metadata and orchestrates task state tracking.
- **Docker Compose**: Containerizes Airflow and MySQL services, isolating them in consistent environments for streamlined management.
- **DEAP**: Performs evolutionary optimization for hyperparameters, classifier selection, and ensemble scoring.
- **Dataset**: IEEE-CIS Fraud Detection data, containing transaction and identity features with fraud labels.


### Pipeline Overview
1. **Data Ingestion**: Load transaction and identity data using TFX `ExampleGen`.
2. **Preprocessing**: Apply feature engineering and data transformations using `Transform`.
3. **Classifier Optimization**:
   - Use DEAP to optimize a population of classifiers, maintaining diversity through mutation.
   - Evaluate individual classifiers based on fraud detection performance.
4. **Ensemble Scoring**:
   - Aggregate predictions from classifiers to produce fraud scores.
   - Use DEAP again to optimize weights for combining classifiers.
5. **Model Training**: Train the best-performing ensemble and evaluate its fraud detection performance.
6. **Deployment**: Export the trained model for deployment in production.

---

## Why This Approach?
Fraud detection demands a system capable of adapting to evolving patterns and edge cases. By leveraging evolutionary optimization with DEAP, this project aims to ensure:
- **Robustness**: Diverse classifiers mitigate the risk of overfitting to specific fraud patterns.
- **Accuracy**: Optimized ensemble scoring improves fraud detection metrics like F1-score and AUC.
- **Scalability**: The pipeline supports large datasets and adapts to changing fraud behavior.

---

*"Card denied, sir. Would you like to try another one?"*
