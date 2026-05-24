import json
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from traffic_prediction.config.settings import load_config
from traffic_prediction.data.processor import DataProcessor
from traffic_prediction.models.lstm import LSTMModelConfig
from traffic_prediction.training.trainer import LSTMTrainer, TrainingLoopConfig
from traffic_prediction.models.registry import ModelRegistry

def main():
    config = load_config()
    print("Starting Grid Search...")
    
    # Get latest featured dataset
    reports_dir = config.paths.reports_dir
    report_dirs = [d for d in reports_dir.iterdir() if d.is_dir() and "offline" in d.name]
    if not report_dirs:
        print("No offline pipeline reports found. Please run offline pipeline first.")
        return
    latest_report = max(report_dirs, key=os.path.getmtime)
    featured_path = latest_report / "featured_dataset.pkl"
    print(f"Loading dataset from: {featured_path}")
    
    featured = pd.read_pickle(featured_path)
    
    processor = DataProcessor(config.data, config.features)
    
    # Feature columns (excluding typical non-features)
    DEFAULT_EXCLUDED_FEATURE_COLUMNS = {
        "id", "road_id", "road_name", "city", "collected_at_wib", "frc"
    }
    numeric_columns = featured.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in DEFAULT_EXCLUDED_FEATURE_COLUMNS]
    
    print("Splitting data...")
    train, validation, test, _ = processor.chronological_split(featured)
    train_scaled = processor.fit_transform_train(train, feature_columns)
    validation_scaled = processor.transform_eval(validation)
    
    print("Creating sequences...")
    x_train, y_train, _ = processor.create_sequences(train_scaled, feature_columns)
    x_validation, y_validation, _ = processor.create_sequences(validation_scaled, feature_columns)
    
    # Grid Search Definition
    param_grid = {
        "hidden_sizes": [(128, 64), (64, 32)],
        "dropout": [0.3, 0.4, 0.5],
        "learning_rate": [0.001, 0.0003]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    registry = ModelRegistry(config.paths.models_dir / "registry.json")
    
    results = []
    
    for i, combination in enumerate(combinations, 1):
        params = dict(zip(keys, combination))
        print(f"\n[{i}/{len(combinations)}] Testing configuration: {params}")
        
        model_config = LSTMModelConfig(
            input_size=len(feature_columns),
            prediction_horizon=4,
            hidden_sizes=params["hidden_sizes"],
            dense_units=32,
            dropout=params["dropout"],
            seq2seq=True,
        )
        
        training_config = TrainingLoopConfig(
            max_epochs=150,  # Increased from 40 to allow full convergence for lower learning rates
            batch_size=64,
            learning_rate=params["learning_rate"],
            early_stopping_patience=15,
        )
        
        trainer = LSTMTrainer(model_config=model_config, training_config=training_config)
        
        try:
            result = trainer.train(
                X_train=x_train,
                y_train=y_train,
                X_validation=x_validation,
                y_validation=y_validation,
                artifact_root=config.paths.reports_dir,
                registry=registry,
                extra_metadata={"model_type": "seq2seq_lstm", "grid_search": True, **params}
            )
            
            results.append({
                "params": params,
                "model_version": result.model_version,
                "best_epoch": result.best_epoch,
                "validation_mae": result.validation_mae,
                "validation_rmse": result.validation_rmse
            })
            print(f"Result -> MAE: {result.validation_mae:.4f}, Epoch: {result.best_epoch}")
        except Exception as e:
            print(f"Training failed for config {params}: {e}")
    
    # Save grid search results
    gs_results_path = config.paths.reports_dir / f"grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    gs_results_path.write_text(json.dumps(results, indent=2))
    
    print(f"\nGrid search completed! Results saved to {gs_results_path}")
    
    # Print Top 3
    results.sort(key=lambda x: x["validation_mae"])
    print("\nTop 3 Configurations:")
    for i, res in enumerate(results[:3], 1):
        print(f"#{i}: MAE={res['validation_mae']:.4f} | Params: {res['params']}")

if __name__ == "__main__":
    main()
