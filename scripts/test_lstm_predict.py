"""
Test end-to-end: apakah LSTM bisa menghasilkan prediksi dari live buffer?
Jalankan: python scripts/test_lstm_predict.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load env vars dari .env jika ada
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from traffic_prediction.config.settings import load_config
from traffic_prediction.persistence.postgresql import PostgreSQLPersistence as db_cls
from traffic_prediction.ingestion.buffer import LiveBufferManager

def main():
    config = load_config()
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DATABASE_URL")
    
    if not db_url:
        print("WARNING: No DATABASE_URL found. Cannot test with live buffer.")
        print("Set DATABASE_URL in .env or env to run this test.")
        return
    
    print(f"Connecting to DB...")
    try:
        db = db_cls(db_url)
        records = db.latest_live_records(limit=5000)
        print(f"Loaded {len(records)} records from DB")
    except Exception as e:
        print(f"ERROR connecting to DB: {e}")
        return
    
    # Seed buffer
    buffer = LiveBufferManager(min_timesteps=12, max_timesteps=48)
    buffer.append_many(records)
    
    road_ids = list(buffer.buffers.keys())
    print(f"\nBuffer seeded for {len(road_ids)} roads")
    
    for road_id in road_ids[:3]:
        buf_len = len(buffer.buffers[road_id])
        has_min = buffer.has_minimum_history(road_id)
        latest = buffer.get_latest(road_id, n=1)
        latest_speed = latest[-1].current_speed if latest else None
        print(f"  {road_id}: {buf_len} timesteps, has_min={has_min}, latest_speed={latest_speed}")
    
    # Test model prediction on first road with enough data
    from traffic_prediction.inference.runner import PyTorchModelRunner
    from traffic_prediction.artifacts import ArtifactLayout
    from traffic_prediction.features.online import OnlineFeatureEngineer
    from traffic_prediction.data.schemas import FeatureManifest
    import json, pandas as pd
    
    models_dir = config.paths.models_dir
    model_path = models_dir / "lstm-real-20260518-032721"
    
    runner = PyTorchModelRunner.load_from_artifact(model_path)
    
    # Load manifest
    cfg_path = model_path / "model_config.json"
    payload = json.loads(cfg_path.read_text())
    fm_dict = payload["extra_metadata"]["feature_manifest"]
    manifest = FeatureManifest(**fm_dict)
    
    # Load scaler
    import joblib
    scaler_path = model_path / "scaler_params.joblib"
    scaler_store = joblib.load(scaler_path) if scaler_path.exists() else None
    print(f"\nScaler loaded: {scaler_store is not None}")
    
    # Load roads
    roads_path = config.paths.roads_master_csv
    if roads_path.exists():
        roads = pd.read_csv(roads_path)
        print(f"Roads loaded: {len(roads)} roads")
    else:
        print(f"WARNING: roads master CSV not found at {roads_path}")
        return
    
    # Build online feature engineer
    try:
        from traffic_prediction.features.spatial import build_neighbor_mapping
        neighbor_mapping = build_neighbor_mapping(roads, config.features.spatial_neighbor_count)
    except Exception:
        neighbor_mapping = {}
    
    ffe = OnlineFeatureEngineer(
        manifest=manifest,
        buffer_manager=buffer,
        roads=roads,
        scaler_store=scaler_store,
        neighbor_mapping=neighbor_mapping,
    )
    
    # Test predict on road that has enough history
    for road_id in road_ids:
        if buffer.has_minimum_history(road_id):
            print(f"\nTesting prediction for road: {road_id}")
            try:
                feature_result = ffe.build_features(road_id)
                print(f"  Feature result: has_min={feature_result.has_minimum_history}, quality={feature_result.quality.status}")
                if feature_result.has_minimum_history and feature_result.sequence is not None:
                    preds = runner.predict_kmh(feature_result.sequence)
                    print(f"  LSTM Predictions: {[f'{p:.1f}' for p in preds]} km/h")
                    print(f"  ==> LSTM PREDICTION WORKS!")
                else:
                    print(f"  Insufficient features: {feature_result.quality}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            break
    else:
        print("\nNo road has minimum history (12 timesteps). LSTM cannot run.")
        print("Need more live data in buffer.")

if __name__ == "__main__":
    main()
