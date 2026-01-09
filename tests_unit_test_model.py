import pytest
import pandas as pd
import numpy as np
from src.train import preprocess_data, train_model
from src.monitoring.drift_detector import ModelDriftDetector

class TestDataPreprocessing:
    def test_preprocess_data(self):
        # Create sample data
        df = pd.DataFrame({
            'Class': [0, 1, 0],
            'V1': [1.0, 2.0, 3.0],
            'Amount': [100, 200, 300]
        })

        X, y = preprocess_data(df)

        assert X.shape[0] == 3
        assert y.shape[0] == 3
        assert 'Class' not in X.columns

    def test_missing_values_handling(self):
        df = pd.DataFrame({
            'Class': [0, 1, 0],
            'V1': [1.0, np.nan, 3.0],
            'Amount': [100, 200, 300]
        })

        X, y = preprocess_data(df)

        assert X.isna().sum().sum() == 0

class TestModel:
    def test_model_training(self):
        # Create synthetic training data
        X_train = np.random.rand(100, 30)
        y_train = np.random.randint(0, 2, 100)

        model = train_model(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_model_prediction_shape(self):
        X_train = np.random.rand(100, 30)
        y_train = np.random.randint(0, 2, 100)
        model = train_model(X_train, y_train)

        X_test = np.random.rand(10, 30)
        predictions = model.predict(X_test)

        assert predictions.shape[0] == 10
        assert all(p in [0, 1] for p in predictions)

class TestDriftDetection:
    def test_drift_detector_initialization(self):
        baseline_data = pd.DataFrame({
            f'V{i}': np.random.rand(100) for i in range(1, 29)
        })
        baseline_data['Amount'] = np.random.rand(100) * 1000
        baseline_data.to_csv('baseline_test.csv', index=False)

        detector = ModelDriftDetector('baseline_test.csv', threshold=0.05)

        assert detector.threshold == 0.05
        assert len(detector.feature_columns) == 29

    def test_no_drift_detection(self):
        # Create identical distributions
        baseline_data = pd.DataFrame({
            f'V{i}': np.random.rand(1000) for i in range(1, 29)
        })
        baseline_data['Amount'] = np.random.rand(1000) * 1000
        baseline_data.to_csv('baseline_test2.csv', index=False)

        detector = ModelDriftDetector('baseline_test2.csv', threshold=0.05)

        # Test with same distribution
        current_data = baseline_data.copy()
        result = detector.detect_data_drift(current_data)

        assert result['drift_detected'] == False
