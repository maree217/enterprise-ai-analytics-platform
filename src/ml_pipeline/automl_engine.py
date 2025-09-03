"""
AutoML Engine - Automated Machine Learning Pipeline

This module provides automated machine learning capabilities including
model training, hyperparameter optimization, and model deployment.

Author: Ram Senthil-Maree
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib

from ..data_pipeline.feature_store import FeatureStore
from ..deployment.model_registry import ModelRegistry
from ..deployment.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    r2_score: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0


@dataclass
class TrainedModel:
    """Trained model container"""
    model: Any
    algorithm: str
    performance: ModelPerformance
    features: List[str]
    target: str
    model_type: str  # 'regression' or 'classification'
    hyperparameters: Dict[str, Any]
    training_data_shape: Tuple[int, int]
    created_at: datetime


class AutoMLEngine:
    """
    Automated Machine Learning Engine
    
    Provides end-to-end automated machine learning including:
    - Feature engineering and selection
    - Algorithm selection and training
    - Hyperparameter optimization
    - Model evaluation and selection
    - Model deployment and monitoring
    """
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_store = FeatureStore(config.feature_store_config)
        self.model_registry = ModelRegistry(config.model_registry_config)
        self.trained_models: Dict[str, TrainedModel] = {}
        self.is_monitoring = False
        
        # Supported algorithms
        self.regression_algorithms = {
            'random_forest': RandomForestRegressor,
            'xgboost': XGBRegressor,
            'lightgbm': LGBMRegressor
        }
        
        self.classification_algorithms = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }
    
    async def health_check(self) -> bool:
        """Check if the AutoML engine is healthy"""
        try:
            # Check feature store connection
            await self.feature_store.health_check()
            
            # Check model registry connection
            await self.model_registry.health_check()
            
            logger.info("AutoML Engine health check passed")
            return True
            
        except Exception as e:
            logger.error(f"AutoML Engine health check failed: {e}")
            return False
    
    async def train_models(self, dataset_id: str, target_variable: str, 
                          model_type: str = "auto") -> Dict[str, TrainedModel]:
        """
        Train multiple models automatically
        
        Args:
            dataset_id: Identifier for the dataset
            target_variable: Name of the target variable
            model_type: 'regression', 'classification', or 'auto'
        
        Returns:
            Dictionary of trained models with performance metrics
        """
        logger.info(f"Starting AutoML training for dataset {dataset_id}")
        
        try:
            # Load and prepare data
            data = await self.feature_store.get_dataset(dataset_id)
            features, target = await self._prepare_data(data, target_variable)
            
            # Determine model type automatically if not specified
            if model_type == "auto":
                model_type = self._detect_model_type(target)
            
            # Feature engineering
            engineered_features = await self._auto_feature_engineering(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                engineered_features, target, test_size=0.2, random_state=42
            )
            
            # Select algorithms based on model type
            algorithms = (self.regression_algorithms if model_type == 'regression' 
                         else self.classification_algorithms)
            
            trained_models = {}
            
            # Train each algorithm
            for algo_name, algo_class in algorithms.items():
                logger.info(f"Training {algo_name} model...")
                
                model = await self._train_single_model(
                    algo_class, X_train, y_train, X_test, y_test, 
                    algo_name, model_type, target_variable
                )
                
                trained_models[algo_name] = model
                self.trained_models[f"{dataset_id}_{algo_name}"] = model
            
            # Train ensemble model
            ensemble_model = await self._train_ensemble_model(
                trained_models, X_train, y_train, X_test, y_test,
                model_type, target_variable
            )
            
            trained_models['ensemble'] = ensemble_model
            self.trained_models[f"{dataset_id}_ensemble"] = ensemble_model
            
            # Select best model
            best_model = self._select_best_model(trained_models, model_type)
            
            # Register best model
            await self.model_registry.register_model(
                model=best_model.model,
                name=f"{dataset_id}_{target_variable}_best",
                metrics=best_model.performance.__dict__,
                features=best_model.features,
                algorithm=best_model.algorithm
            )
            
            logger.info(f"AutoML training completed. Best model: {best_model.algorithm}")
            
            return trained_models
            
        except Exception as e:
            logger.error(f"AutoML training failed: {e}")
            raise
    
    async def _prepare_data(self, data: pd.DataFrame, 
                           target_variable: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Separate features and target
        features = data.drop(columns=[target_variable])
        target = data[target_variable]
        
        # Handle missing values
        features = features.fillna(features.mean() if features.select_dtypes(include=[np.number]).shape[1] > 0 else features.mode().iloc[0])
        target = target.fillna(target.mean() if pd.api.types.is_numeric_dtype(target) else target.mode()[0])
        
        # Encode categorical variables
        categorical_columns = features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            features[col] = pd.Categorical(features[col]).codes
        
        return features, target
    
    def _detect_model_type(self, target: pd.Series) -> str:
        """Automatically detect if problem is regression or classification"""
        if pd.api.types.is_numeric_dtype(target):
            # Check if target has few unique values (likely classification)
            unique_values = target.nunique()
            total_values = len(target)
            
            if unique_values <= 10 or unique_values / total_values < 0.05:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    async def _auto_feature_engineering(self, features: pd.DataFrame) -> pd.DataFrame:
        """Automated feature engineering"""
        engineered_features = features.copy()
        
        # Numerical features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if not numeric_features.empty:
            # Create interaction features for top correlated pairs
            correlations = numeric_features.corr().abs()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    if 0.3 < correlations.iloc[i, j] < 0.9:  # Avoid perfect correlation
                        high_corr_pairs.append((correlations.columns[i], correlations.columns[j]))
            
            # Create interaction features (limit to top 10)
            for i, (feat1, feat2) in enumerate(high_corr_pairs[:10]):
                engineered_features[f'{feat1}_x_{feat2}'] = features[feat1] * features[feat2]
                engineered_features[f'{feat1}_div_{feat2}'] = (features[feat1] / (features[feat2] + 1e-8))
            
            # Create polynomial features for top features
            feature_importance = numeric_features.var().sort_values(ascending=False)
            top_features = feature_importance.head(5).index
            
            for feature in top_features:
                engineered_features[f'{feature}_squared'] = features[feature] ** 2
                engineered_features[f'{feature}_log'] = np.log1p(np.abs(features[feature]))
        
        return engineered_features
    
    async def _train_single_model(self, algo_class, X_train: pd.DataFrame, 
                                 y_train: pd.Series, X_test: pd.DataFrame, 
                                 y_test: pd.Series, algo_name: str, 
                                 model_type: str, target_variable: str) -> TrainedModel:
        """Train a single model with hyperparameter optimization"""
        start_time = datetime.now()
        
        # Basic hyperparameters (in production, use more sophisticated optimization)
        if algo_name == 'random_forest':
            params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        elif algo_name == 'xgboost':
            params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
        elif algo_name == 'lightgbm':
            params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1}
        else:
            params = {}
        
        # Train model
        model = algo_class(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Measure inference time
        inference_start = datetime.now()
        _ = model.predict(X_test[:10] if len(X_test) > 10 else X_test)
        inference_time = (datetime.now() - inference_start).total_seconds() / min(10, len(X_test))
        
        if model_type == 'regression':
            performance = ModelPerformance(
                r2_score=model.score(X_test, y_test),
                rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                mae=np.mean(np.abs(y_test - y_pred)),
                training_time=training_time,
                inference_time=inference_time
            )
        else:
            performance = ModelPerformance(
                accuracy=accuracy_score(y_test, y_pred),
                training_time=training_time,
                inference_time=inference_time
            )
            
            # Add precision, recall, f1 for classification
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                performance.precision = precision_score(y_test, y_pred, average='weighted')
                performance.recall = recall_score(y_test, y_pred, average='weighted') 
                performance.f1_score = f1_score(y_test, y_pred, average='weighted')
            except Exception:
                pass  # Skip if not applicable
        
        return TrainedModel(
            model=model,
            algorithm=algo_name,
            performance=performance,
            features=list(X_train.columns),
            target=target_variable,
            model_type=model_type,
            hyperparameters=params,
            training_data_shape=X_train.shape,
            created_at=datetime.now()
        )
    
    async def _train_ensemble_model(self, individual_models: Dict[str, TrainedModel],
                                   X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   model_type: str, target_variable: str) -> TrainedModel:
        """Train an ensemble model combining individual models"""
        start_time = datetime.now()
        
        # Simple voting/averaging ensemble
        predictions = []
        for model_name, trained_model in individual_models.items():
            pred = trained_model.model.predict(X_test)
            predictions.append(pred)
        
        # Average predictions for regression, majority vote for classification
        if model_type == 'regression':
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            # Simple majority voting
            ensemble_pred = []
            predictions_array = np.array(predictions)
            for i in range(len(X_test)):
                votes = predictions_array[:, i]
                ensemble_pred.append(max(set(votes), key=list(votes).count))
            ensemble_pred = np.array(ensemble_pred)
        
        # Calculate performance
        training_time = (datetime.now() - start_time).total_seconds()
        
        if model_type == 'regression':
            performance = ModelPerformance(
                r2_score=1 - (np.sum((y_test - ensemble_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)),
                rmse=np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                mae=np.mean(np.abs(y_test - ensemble_pred)),
                training_time=training_time,
                inference_time=sum(m.performance.inference_time for m in individual_models.values())
            )
        else:
            performance = ModelPerformance(
                accuracy=accuracy_score(y_test, ensemble_pred),
                training_time=training_time,
                inference_time=sum(m.performance.inference_time for m in individual_models.values())
            )
        
        # Create ensemble "model" (just a collection of individual models)
        ensemble_model_data = {
            'models': {name: model.model for name, model in individual_models.items()},
            'model_type': model_type
        }
        
        return TrainedModel(
            model=ensemble_model_data,
            algorithm='ensemble',
            performance=performance,
            features=list(X_train.columns),
            target=target_variable,
            model_type=model_type,
            hyperparameters={'ensemble_method': 'averaging' if model_type == 'regression' else 'voting'},
            training_data_shape=X_train.shape,
            created_at=datetime.now()
        )
    
    def _select_best_model(self, models: Dict[str, TrainedModel], model_type: str) -> TrainedModel:
        """Select the best performing model"""
        if model_type == 'regression':
            # Select model with highest R² score
            best_model = max(models.values(), key=lambda m: m.performance.r2_score)
        else:
            # Select model with highest accuracy
            best_model = max(models.values(), key=lambda m: m.performance.accuracy)
        
        logger.info(f"Best model selected: {best_model.algorithm}")
        return best_model
    
    async def predict(self, model_id: str, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.trained_models[model_id]
        
        if model.algorithm == 'ensemble':
            # Handle ensemble prediction
            predictions = []
            for individual_model in model.model['models'].values():
                pred = individual_model.predict(features)
                predictions.append(pred)
            
            if model.model_type == 'regression':
                return np.mean(predictions, axis=0)
            else:
                # Majority voting
                ensemble_pred = []
                predictions_array = np.array(predictions)
                for i in range(len(features)):
                    votes = predictions_array[:, i]
                    ensemble_pred.append(max(set(votes), key=list(votes).count))
                return np.array(ensemble_pred)
        else:
            return model.model.predict(features)
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.trained_models[model_id]
        
        return {
            'algorithm': model.algorithm,
            'model_type': model.model_type,
            'performance': model.performance.__dict__,
            'features': model.features,
            'target': model.target,
            'hyperparameters': model.hyperparameters,
            'training_data_shape': model.training_data_shape,
            'created_at': model.created_at.isoformat()
        }
    
    async def start_model_monitoring(self):
        """Start background model monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            asyncio.create_task(self._model_monitoring_loop())
            logger.info("Model monitoring started")
    
    async def stop_model_monitoring(self):
        """Stop model monitoring"""
        self.is_monitoring = False
        logger.info("Model monitoring stopped")
    
    async def _model_monitoring_loop(self):
        """Background task for monitoring model performance"""
        while self.is_monitoring:
            try:
                # Monitor each trained model
                for model_id, model in self.trained_models.items():
                    # In production, this would check for data drift, performance degradation, etc.
                    logger.debug(f"Monitoring model {model_id}")
                    
                    # Example: Check if model is older than 30 days (retrain trigger)
                    if (datetime.now() - model.created_at) > timedelta(days=30):
                        logger.warning(f"Model {model_id} is over 30 days old - consider retraining")
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in model monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_model_monitoring()
        logger.info("AutoML Engine cleanup completed")