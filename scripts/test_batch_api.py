import urllib.request, json

url = 'https://awai-backend.onrender.com/predict/batch'
payload = json.dumps({
    'predictions': [
        {'road_id': 'JKT001', 'horizon_minutes': 15},
        {'road_id': 'JKT002', 'horizon_minutes': 15},
        {'road_id': 'JKT003', 'horizon_minutes': 15}
    ]
}).encode()

req = urllib.request.Request(url, data=payload, headers={
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'x-api-key': 'awai_api_key_rev2026'
}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=35) as r:
        data = json.loads(r.read())
        print('successful_count:', data.get('successful_count'))
        print('failed_count:', data.get('failed_count'))
        for p in data.get('predictions', [])[:3]:
            rid = p.get('road_id')
            spd = p.get('predicted_speed')
            mth = p.get('prediction_method')
            conf = p.get('confidence_score')
            print(f'  {rid}: {spd:.1f} km/h, method={mth}, conf={conf}')
except Exception as e:
    print('Error:', type(e).__name__, str(e)[:300])
