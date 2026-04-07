#!/usr/bin/env python
"""
Test script for Bright Data residential proxy
"""
import sys
import requests
import os
import json
from datetime import datetime
import urllib3

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_brightdata_proxy():
    """Test Bright Data residential proxy"""
    print("Testing Bright Data residential proxy...")
    
    # Proxy configuration
    proxy = "http://brd-customer-hl_c7bff232-zone-residential_proxy1-country-us:w46vs0z46xmc@brd.superproxy.io:33335"
    
    # Setup proxies for requests
    proxies = {
        "http": proxy,
        "https": proxy
    }
    
    try:
        # First, test with the Bright Data test endpoint
        response = requests.get(
            "https://geo.brdtest.com/welcome.txt?product=resi&method=native",
            proxies=proxies,
            timeout=10,
            verify=False  # Disable SSL verification
        )
        
        if response.status_code == 200:
            print("✅ Bright Data test successful!")
            print(f"Response: {response.text.strip()}")
        else:
            print(f"❌ Bright Data test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        
        # Then test with Instagram
        print("\nTesting access to Instagram...")
        instagram_response = requests.get(
            "https://www.instagram.com/favicon.ico",
            proxies=proxies,
            timeout=10,
            verify=False,  # Disable SSL verification
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        
        if instagram_response.status_code == 200:
            print("✅ Instagram test successful!")
            print(f"Status code: {instagram_response.status_code}")
            print(f"Content length: {len(instagram_response.content)} bytes")
        else:
            print(f"❌ Instagram test failed with status code: {instagram_response.status_code}")
            
        # Save test results
        results = {
            "timestamp": datetime.now().isoformat(),
            "brightdata_test": {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_text": response.text.strip()
            },
            "instagram_test": {
                "status_code": instagram_response.status_code,
                "success": instagram_response.status_code == 200,
                "content_length": len(instagram_response.content)
            }
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join("data", "logs"), exist_ok=True)
        
        # Save results to a file
        with open(os.path.join("data", "logs", "proxy_test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nTest results saved to data/logs/proxy_test_results.json")
        
    except Exception as e:
        print(f"❌ Error testing proxy: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_brightdata_proxy() 