import requests
import json
import time

# --- CONFIGURATION ---
URL = "http://127.0.0.1:5000/predict"
PING_URL = "http://127.0.0.1:5000/ping"

# ------------------------------
# 0. Test server health
# ------------------------------
def test_ping():
    print("\n" + "="*60)
    print("TESTING SERVER HEALTH")
    print("="*60)
    try:
        r = requests.get(PING_URL, timeout=5)
        print("Status:", r.status_code)
        if r.status_code == 200:
            print("Response:", json.dumps(r.json(), indent=2))
            return True
        else:
            print("Server not healthy!")
            return False
    except Exception as e:
        print(f"ERROR: Cannot connect to server - {e}")
        return False

# ------------------------------
# 1. Existing Test Cases
# ------------------------------

# Case A: Raw HTML (Validation team's format)
html_payload = {
    "html": '<p content="Hello NeoDev!" style="font-size: 16px; font-weight: normal;">Hello NeoDev!</p>'
}

# Case B: JSON DOM Payload (Standard)
json_payload = {
    "dom": {
        "tag": "div",
        "attributes": {"class": "text-content"},
        "children": [
            {"tag": "p", "attributes": {}, "children": []}
        ]
    }
}

# ------------------------------
# 2. NEW HIERARCHICAL TEST CASES
# ------------------------------

# Case C: The "Hero" Section (Complex Hierarchy)
# Model should see: Section -> (H1 + P + Button)
hero_payload = {
    "dom": {
        "tag": "section",
        "attributes": {"class": "hero-section full-width"},
        "children": [
            {"tag": "h1", "attributes": {"class": "hero-title"}, "children": []},
            {"tag": "p", "attributes": {"class": "subtext"}, "children": []},
            {"tag": "button", "attributes": {"class": "cta-button"}, "children": []}
        ]
    }
}

# Case D: The "Button" Component (Simple Hierarchy)
# Model should see: Button -> Span (Icon/Text)
button_payload = {
    "dom": {
        "tag": "button",
        "attributes": {"class": "btn btn-primary login-submit", "type": "submit"},
        "children": [
            {"tag": "span", "attributes": {"class": "icon-user"}, "children": []}
        ]
    }
}

# ------------------------------
# Test Runner
# ------------------------------
def test(payload, description):
    print(f"\n--- Testing: {description} ---")
    try:
        start = time.time()
        r = requests.post(URL, json=payload, timeout=10)
        latency = (time.time() - start) * 1000
        
        if r.status_code == 200:
            res = r.json()
            label = res.get("predicted_label", "Unknown")
            conf = res.get("confidence", 0.0)
            
            # Print Result clearly
            print(f"Status: 200 OK ({latency:.1f}ms)")
            print(f"Prediction: {label}")
            print(f"Confidence: {conf:.2%}")
            # print(f"Full Response: {res}") # Uncomment for debugging
        else:
            print(f"Error {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" VALIDATION SUITE ")
    print("="*60)
    
    if test_ping():
        # Run existing tests
        test(html_payload, "Raw HTML Input")
        test(json_payload, "Basic JSON Input")
        
        # Run NEW Hierarchical tests
        test(hero_payload, "Hero Section (Complex Structure)")
        test(button_payload, "Button Component (Nested Structure)")
        
    print("\n" + "="*60)
    print(" SUITE COMPLETE ")
    print("="*60)