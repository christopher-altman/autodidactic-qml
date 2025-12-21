
import os
import json
import pytest
import sys
from datetime import datetime
from unittest.mock import patch
import experiments.kt2_locality_falsifier as subject

def test_decisive_run_provenance():
    """Verify that the decisive run produces a JSON artifact with required metadata."""
    
    # Use a specific output file for testing
    test_output_dir = "tests/tmp_results"
    expected_file = os.path.join(test_output_dir, "kt2_decisive_1step.json")
    
    # Clean up previous
    if os.path.exists(expected_file):
        os.remove(expected_file)
        
    # Mock sys.argv to simulate CLI call
    test_args = ["experiments.kt2_locality_falsifier", "--run-decisive", "--output-dir", test_output_dir, "--quiet"]
    
    with patch.object(sys, 'argv', test_args):
        subject.main()
        
    assert os.path.exists(expected_file), "Decisive artifact was not created."
    
    with open(expected_file, 'r') as f:
        data = json.load(f)
        
    # Check Top Level Meta
    assert "meta" in data, "Provenance metadata missing."
    meta = data["meta"]
    
    # Check Timestamp Format (fail-soft ISO check)
    ts = meta.get("timestamp_utc")
    assert ts and ts.endswith("Z"), "Timestamp must be UTC Z-suffixed"
    try:
        # Python 3.10+ fromisoformat handles 'Z' often, but be safe: replace Z with +00:00
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        pytest.fail(f"timestamp_utc {ts} is not valid ISO-8601")

    # Check Required Fields
    assert "command" in meta
    assert "repo" in meta
    assert "platform" in meta
    assert meta["platform"]["python"] is not None
    assert "dependencies" in meta
    assert "seeds" in meta
    
    # Check Run Params (Focus 4)
    run_params = meta.get("run_params", {})
    assert "ci_threshold" in run_params
    assert run_params["ci_threshold"] == 0.10
    # Austere check: flattened params
    assert run_params["dim"] == 12
    assert run_params["lr"] == 1e-3
    assert run_params["k"] == 1
    
    # Check Determinism
    seeds = meta["seeds"]
    assert seeds["PERTURB"] == 42
    assert seeds["EVAL"] == 12345
    assert seeds["RECOVERY"] == 2025
    
    # Check Scientific Integrity (did we break the result?)
    assert "verdict" in data
    assert "decisive_table" in data
    
    # Clean up
    import shutil
    shutil.rmtree(test_output_dir)
