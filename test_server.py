"""
Test if server is running and accessible
"""
import requests

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✅ Server is running!")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("❌ Server is NOT running!")
    print("Please start the server with: cd backend && python main.py")
except Exception as e:
    print(f"❌ Error: {e}")
