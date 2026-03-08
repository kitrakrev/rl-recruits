import requests
import json
import sys

def test_env():
    base_url = "http://localhost:8000"
    print(f"Testing environment at {base_url}...")
    
    # 1. Health
    requests.get(f"{base_url}/health")

    # 2. Reset
    print("\n--- RESET ---")
    resp = requests.post(f"{base_url}/reset", json={"seed": 42})
    print(f"Reset Payload: {json.dumps(resp.json(), indent=2)}")

    # 3. Step
    print("\n--- STEP (get_agency_state) ---")
    action = {
        "type": "call_tool",
        "tool_name": "get_agency_state",
        "arguments": {}
    }
    resp = requests.post(f"{base_url}/step", json={"action": action})
    print(f"Step Payload: {json.dumps(resp.json(), indent=2)}")

if __name__ == "__main__":
    test_env()
