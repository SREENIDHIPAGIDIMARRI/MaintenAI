import sys
sys.path.insert(0, '.')
from src.maintenance_recommendations import evaluate_maintenance_rules
from src.failure_prediction import predict_machine_status
from utils.helpers import PRESET_MACHINES

for name, sample in PRESET_MACHINES.items():
    pred = predict_machine_status(sample)
    rec = evaluate_maintenance_rules(sample, pred['failure_probability'], pred['status_label'], pred['risk_category'])
    assert 'subsystem_diagnostics' in rec
    print(f"Scenario: {name}")
    print(f"  Priority: {rec['priority_level']} | Probability: {rec['failure_probability_pct']}%")
    print(f"  Subsystems ({len(rec['subsystem_diagnostics'])}):")
    for s in rec['subsystem_diagnostics']:
        print(f"    - {s['subsystem']}: [{s['status']}] -> {s['action'][:40]}...")

print("\nALL PRESETS VERIFIED WITH 0 ERRORS!")
