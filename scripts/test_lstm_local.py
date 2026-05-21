"""
Quick test: apakah LSTM model bisa load dan prediksi berhasil?
Jalankan: python -m scripts.test_lstm_local
"""
import sys
import json
from pathlib import Path

# Set project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from traffic_prediction.models.registry import ModelRegistry
from traffic_prediction.artifacts import ArtifactLayout
from traffic_prediction.config.settings import load_config

def main():
    config = load_config()
    registry_path = config.paths.models_dir / "registry.json"
    
    print(f"Models dir: {config.paths.models_dir}")
    print(f"Registry path: {registry_path} exists={registry_path.exists()}")
    
    # Try resolving model
    layout = ArtifactLayout.from_paths(config.paths)
    model_path = layout.resolve_latest_model(registry_path)
    print(f"\nResolved model path: {model_path}")
    
    if model_path is None:
        print("ERROR: No model found!")
        return
    
    print(f"Model dir exists: {model_path.exists()}")
    print(f"model.pt exists: {(model_path / 'model.pt').exists()}")
    print(f"model_config.json exists: {(model_path / 'model_config.json').exists()}")
    print(f"scaler_params.joblib exists: {(model_path / 'scaler_params.joblib').exists()}")
    
    # Try loading PyTorchModelRunner
    try:
        from traffic_prediction.inference.runner import PyTorchModelRunner
        runner = PyTorchModelRunner.load_from_artifact(model_path)
        print(f"\nModel runner loaded: {runner}")
        print(f"Model type: {type(runner)}")
    except Exception as e:
        print(f"ERROR loading model runner: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try loading feature manifest
    from traffic_prediction.api.app import AppState
    # Simpler check
    try:
        manifest_path = model_path / "feature_manifest.json"
        if manifest_path.exists():
            from traffic_prediction.data.schemas import FeatureManifest
            manifest = FeatureManifest(**json.loads(manifest_path.read_text()))
            print(f"\nFeature manifest loaded: {manifest.feature_version}, lookback={manifest.lookback}, horizon={manifest.horizon}")
        else:
            # Try from model_config.json
            cfg_path = model_path / "model_config.json"
            if cfg_path.exists():
                payload = json.loads(cfg_path.read_text())
                fm = payload.get("extra_metadata", {}).get("feature_manifest")
                if fm:
                    from traffic_prediction.data.schemas import FeatureManifest
                    manifest = FeatureManifest(**fm)
                    print(f"\nFeature manifest (from model_config): {manifest.feature_version}, lookback={manifest.lookback}, horizon={manifest.horizon}, n_features={len(manifest.feature_columns)}")
                else:
                    print("ERROR: No feature_manifest in model_config.json")
            else:
                print("ERROR: No feature_manifest.json or model_config.json")
    except Exception as e:
        print(f"ERROR loading manifest: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== LSTM model setup looks OK! ===")
    print("Check if buffer has enough data for inference...")

if __name__ == "__main__":
    main()
