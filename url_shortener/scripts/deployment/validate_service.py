#!/usr/bin/env python3
"""
Validation script for URL Shortener service.
Tests the live running service to ensure all functionality works correctly.
"""

import sys
import time
import requests
from typing import Dict, Any, Optional
from datetime import datetime


class ServiceValidator:
    """Validates URL shortener service functionality."""
    
    def __init__(self, base_url: str = "http://localhost:9200"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.test_results = []
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    
    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((name, passed))
        print(f"{status} - {name}")
        if details:
            print(f"       {details}")
    
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                is_healthy = (
                    data.get("status") == "healthy" and
                    data.get("database") == "healthy"
                )
                details = f"DB: {data.get('database')}, Cache: {data.get('cache', 'N/A')}"
                self.print_test("Health Check", is_healthy, details)
                return is_healthy
            else:
                self.print_test("Health Check", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_create_short_url(self) -> Optional[str]:
        """Test creating a short URL."""
        try:
            test_url = f"https://example.com/test/{int(time.time())}"
            response = self.session.post(
                f"{self.base_url}/api/shorten",
                json={"url": test_url},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                short_code = data.get("short_code")
                if short_code:
                    self.print_test(
                        "Create Short URL",
                        True,
                        f"Code: {short_code}, URL: {data.get('short_url')}"
                    )
                    return short_code
            
            self.print_test("Create Short URL", False, f"Status: {response.status_code}")
            return None
        except Exception as e:
            self.print_test("Create Short URL", False, f"Error: {str(e)}")
            return None
    
    def test_get_url_info(self, short_code: str) -> bool:
        """Test getting URL information."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/urls/{short_code}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                has_required_fields = all(
                    key in data for key in ["short_code", "original_url", "created_at"]
                )
                details = f"Access count: {data.get('access_count', 0)}"
                self.print_test("Get URL Info", has_required_fields, details)
                return has_required_fields
            else:
                self.print_test("Get URL Info", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Get URL Info", False, f"Error: {str(e)}")
            return False
    
    def test_redirect(self, short_code: str) -> bool:
        """Test URL redirect functionality."""
        try:
            response = self.session.get(
                f"{self.base_url}/{short_code}",
                allow_redirects=False,
                timeout=5
            )
            
            is_redirect = response.status_code == 302
            location = response.headers.get("Location", "")
            self.print_test(
                "URL Redirect",
                is_redirect,
                f"Redirects to: {location[:50]}..." if location else "No Location header"
            )
            return is_redirect
        except Exception as e:
            self.print_test("URL Redirect", False, f"Error: {str(e)}")
            return False
    
    def test_custom_code(self) -> bool:
        """Test custom short code functionality."""
        try:
            custom_code = f"test{int(time.time())}"
            response = self.session.post(
                f"{self.base_url}/api/shorten",
                json={
                    "url": "https://github.com/example/repo",
                    "custom_code": custom_code
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                code_matches = data.get("short_code") == custom_code
                self.print_test("Custom Short Code", code_matches, f"Code: {custom_code}")
                return code_matches
            else:
                self.print_test("Custom Short Code", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Custom Short Code", False, f"Error: {str(e)}")
            return False
    
    def test_duplicate_custom_code(self, existing_code: str) -> bool:
        """Test duplicate custom code rejection."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/shorten",
                json={
                    "url": "https://different-url.com",
                    "custom_code": existing_code
                },
                timeout=5
            )
            
            is_conflict = response.status_code == 409
            self.print_test(
                "Duplicate Code Rejection",
                is_conflict,
                f"Status: {response.status_code} (expected 409)"
            )
            return is_conflict
        except Exception as e:
            self.print_test("Duplicate Code Rejection", False, f"Error: {str(e)}")
            return False
    
    def test_invalid_url(self) -> bool:
        """Test invalid URL rejection."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/shorten",
                json={"url": "not-a-valid-url"},
                timeout=5
            )
            
            is_rejected = response.status_code == 422
            self.print_test(
                "Invalid URL Rejection",
                is_rejected,
                f"Status: {response.status_code} (expected 422)"
            )
            return is_rejected
        except Exception as e:
            self.print_test("Invalid URL Rejection", False, f"Error: {str(e)}")
            return False
    
    def test_nonexistent_code(self) -> bool:
        """Test accessing non-existent short code."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/urls/nonexistent999",
                timeout=5
            )
            
            is_not_found = response.status_code == 404
            self.print_test(
                "Non-existent Code",
                is_not_found,
                f"Status: {response.status_code} (expected 404)"
            )
            return is_not_found
        except Exception as e:
            self.print_test("Non-existent Code", False, f"Error: {str(e)}")
            return False
    
    def test_stats_endpoint(self) -> bool:
        """Test stats endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/api/stats", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                has_stats = "total_urls" in data or "message" in data
                total = data.get("total_urls", "N/A")
                self.print_test("Stats Endpoint", has_stats, f"Total URLs: {total}")
                return has_stats
            else:
                self.print_test("Stats Endpoint", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Stats Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_web_interface(self) -> bool:
        """Test web interface homepage."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            
            is_ok = response.status_code == 200 and "text/html" in response.headers.get("content-type", "")
            self.print_test(
                "Web Interface",
                is_ok,
                f"Content-Type: {response.headers.get('content-type', 'N/A')}"
            )
            return is_ok
        except Exception as e:
            self.print_test("Web Interface", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        self.print_header("URL Shortener Service Validation")
        print(f"Testing service at: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Basic connectivity
        if not self.test_health_check():
            print("\n❌ Health check failed. Service may not be running.")
            print(f"   Make sure the service is accessible at {self.base_url}")
            return False
        
        print()
        
        # Core functionality tests
        short_code = self.test_create_short_url()
        if short_code:
            self.test_get_url_info(short_code)
            self.test_redirect(short_code)
        
        print()
        
        # Advanced functionality tests
        custom_code = f"test{int(time.time())}"
        response = self.session.post(
            f"{self.base_url}/api/shorten",
            json={"url": "https://example.com/custom", "custom_code": custom_code},
            timeout=5
        )
        if response.status_code == 200:
            self.test_duplicate_custom_code(custom_code)
        
        self.test_invalid_url()
        self.test_nonexistent_code()
        
        print()
        
        # Additional endpoints
        self.test_stats_endpoint()
        self.test_web_interface()
        
        # Print summary
        self.print_summary()
        
        # Return overall success
        return all(passed for _, passed in self.test_results)
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.test_results)
        passed = sum(1 for _, p in self.test_results if p)
        failed = total - passed
        
        self.print_header("Test Summary")
        print(f"Total Tests:  {total}")
        print(f"✅ Passed:     {passed}")
        print(f"❌ Failed:     {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\n⚠️  Failed tests:")
            for name, passed in self.test_results:
                if not passed:
                    print(f"   - {name}")
        
        print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate URL Shortener service functionality"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9200",
        help="Base URL of the service (default: http://localhost:9200)"
    )
    
    args = parser.parse_args()
    
    validator = ServiceValidator(args.url)
    
    try:
        success = validator.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {str(e)}")
        sys.exit(3)


if __name__ == "__main__":
    main()

