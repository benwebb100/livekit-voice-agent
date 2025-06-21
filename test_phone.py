#!/usr/bin/env python3
"""Test script for phone integration"""

import requests
import json
import time

BASE_URL = "https://twillio-server-f1fq.onrender.com"

# Check health
print("Checking health...")
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

# Make an outbound call
# phone_number = input("Enter phone number to call (e.g., +1234567890): ")
phone_number = "+919837277206"  # Example phone number, replace with actual

print(f"\nInitiating call to {phone_number}...")
response = requests.post(
    f"{BASE_URL}/twilio/outbound",
    json={"to": phone_number}
)

if response.status_code == 200:
    result = response.json()
    print(f"Call initiated!")
    print(f"Call SID: {result['sid']}")
    print(f"Room Name: {result['room_name']}")
else:
    print(f"Error: {response.text}")