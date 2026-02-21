#!/bin/bash
# Validation script for URL Shortener service
# Tests the live running service to ensure all functionality works correctly

set -e

BASE_URL="${1:-http://localhost:9200}"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

test_result() {
    local name="$1"
    local passed=$2
    local details="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ $passed -eq 1 ]; then
        echo -e "${GREEN}✅ PASS${NC} - $name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAIL${NC} - $name"
        FAILED_TESTS+=("$name")
    fi
    
    if [ -n "$details" ]; then
        echo "       $details"
    fi
}

# Test 1: Health Check
test_health_check() {
    echo "Testing health check..."
    if response=$(curl -s --max-time 5 "$BASE_URL/api/health" 2>/dev/null); then
        status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        db=$(echo "$response" | grep -o '"database":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$status" = "healthy" ] && [ "$db" = "healthy" ]; then
            test_result "Health Check" 1 "DB: $db, Status: $status"
            return 0
        else
            test_result "Health Check" 0 "Status: $status, DB: $db"
            return 1
        fi
    else
        test_result "Health Check" 0 "Service not accessible"
        return 1
    fi
}

# Test 2: Create Short URL
test_create_short_url() {
    echo "Testing create short URL..."
    test_url="https://example.com/test/$(date +%s)"
    
    if response=$(curl -s --max-time 5 -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"$test_url\"}" 2>/dev/null); then
        
        short_code=$(echo "$response" | grep -o '"short_code":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$short_code" ]; then
            test_result "Create Short URL" 1 "Code: $short_code"
            echo "$short_code"  # Return the code for other tests
            return 0
        else
            test_result "Create Short URL" 0 "No short code in response"
            return 1
        fi
    else
        test_result "Create Short URL" 0 "Request failed"
        return 1
    fi
}

# Test 3: Get URL Info
test_get_url_info() {
    local short_code="$1"
    echo "Testing get URL info..."
    
    if response=$(curl -s --max-time 5 "$BASE_URL/api/urls/$short_code" 2>/dev/null); then
        if echo "$response" | grep -q '"short_code"'; then
            access_count=$(echo "$response" | grep -o '"access_count":[0-9]*' | cut -d':' -f2)
            test_result "Get URL Info" 1 "Access count: $access_count"
            return 0
        else
            test_result "Get URL Info" 0 "Missing required fields"
            return 1
        fi
    else
        test_result "Get URL Info" 0 "Request failed"
        return 1
    fi
}

# Test 4: URL Redirect
test_redirect() {
    local short_code="$1"
    echo "Testing URL redirect..."
    
    # Use -v for verbose output and -s for silent (no progress), capture stderr
    if response=$(curl -s --max-time 5 -v "$BASE_URL/$short_code" 2>&1 | grep -E "(< HTTP|< location)"); then
        if echo "$response" | grep -q "HTTP.*302"; then
            location=$(echo "$response" | grep -i "< location:" | cut -d' ' -f3 | tr -d '\r\n')
            test_result "URL Redirect" 1 "Redirects to: ${location:0:50}..."
            return 0
        else
            test_result "URL Redirect" 0 "No 302 redirect found"
            return 1
        fi
    else
        test_result "URL Redirect" 0 "Request failed"
        return 1
    fi
}

# Test 5: Custom Code
test_custom_code() {
    echo "Testing custom short code..."
    custom_code="test$(date +%s)"
    
    if response=$(curl -s --max-time 5 -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"https://github.com/example/repo\",\"custom_code\":\"$custom_code\"}" 2>/dev/null); then
        
        returned_code=$(echo "$response" | grep -o '"short_code":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$returned_code" = "$custom_code" ]; then
            test_result "Custom Short Code" 1 "Code: $custom_code"
            echo "$custom_code"  # Return for duplicate test
            return 0
        else
            test_result "Custom Short Code" 0 "Code mismatch: got $returned_code"
            return 1
        fi
    else
        test_result "Custom Short Code" 0 "Request failed"
        return 1
    fi
}

# Test 6: Duplicate Code Rejection
test_duplicate_code() {
    local existing_code="$1"
    echo "Testing duplicate code rejection..."
    
    if response=$(curl -s --max-time 5 -w "\n%{http_code}" -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"https://different-url.com\",\"custom_code\":\"$existing_code\"}" 2>/dev/null); then
        
        http_code=$(echo "$response" | tail -1)
        
        if [ "$http_code" = "409" ]; then
            test_result "Duplicate Code Rejection" 1 "Status: 409 (expected)"
            return 0
        else
            test_result "Duplicate Code Rejection" 0 "Status: $http_code (expected 409)"
            return 1
        fi
    else
        test_result "Duplicate Code Rejection" 0 "Request failed"
        return 1
    fi
}

# Test 7: Invalid URL Rejection
test_invalid_url() {
    echo "Testing invalid URL rejection..."
    
    if response=$(curl -s --max-time 5 -w "\n%{http_code}" -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d '{"url":"not-a-valid-url"}' 2>/dev/null); then
        
        http_code=$(echo "$response" | tail -1)
        
        if [ "$http_code" = "422" ]; then
            test_result "Invalid URL Rejection" 1 "Status: 422 (expected)"
            return 0
        else
            test_result "Invalid URL Rejection" 0 "Status: $http_code (expected 422)"
            return 1
        fi
    else
        test_result "Invalid URL Rejection" 0 "Request failed"
        return 1
    fi
}

# Test 8: Non-existent Code
test_nonexistent_code() {
    echo "Testing non-existent code..."
    
    if response=$(curl -s --max-time 5 -w "\n%{http_code}" "$BASE_URL/api/urls/nonexistent999" 2>/dev/null); then
        http_code=$(echo "$response" | tail -1)
        
        if [ "$http_code" = "404" ]; then
            test_result "Non-existent Code" 1 "Status: 404 (expected)"
            return 0
        else
            test_result "Non-existent Code" 0 "Status: $http_code (expected 404)"
            return 1
        fi
    else
        test_result "Non-existent Code" 0 "Request failed"
        return 1
    fi
}

# Test 9: Stats Endpoint
test_stats() {
    echo "Testing stats endpoint..."
    
    if response=$(curl -s --max-time 5 "$BASE_URL/api/stats" 2>/dev/null); then
        if echo "$response" | grep -q '"total_urls"'; then
            total=$(echo "$response" | grep -o '"total_urls":[0-9]*' | cut -d':' -f2)
            test_result "Stats Endpoint" 1 "Total URLs: $total"
            return 0
        else
            test_result "Stats Endpoint" 1 "Stats available"
            return 0
        fi
    else
        test_result "Stats Endpoint" 0 "Request failed"
        return 1
    fi
}

# Test 10: Web Interface
test_web_interface() {
    echo "Testing web interface..."
    
    if response=$(curl -s --max-time 5 -w "\n%{http_code}\n%{content_type}" "$BASE_URL/" 2>/dev/null); then
        http_code=$(echo "$response" | tail -2 | head -1)
        content_type=$(echo "$response" | tail -1)
        
        if [ "$http_code" = "200" ] && echo "$content_type" | grep -iq "text/html"; then
            test_result "Web Interface" 1 "Homepage accessible"
            return 0
        else
            test_result "Web Interface" 0 "Status: $http_code, Type: $content_type"
            return 1
        fi
    else
        test_result "Web Interface" 0 "Request failed"
        return 1
    fi
}

# Print summary
print_summary() {
    print_header "Test Summary"
    
    echo "Total Tests:  $TOTAL_TESTS"
    echo -e "${GREEN}✅ Passed:     $PASSED_TESTS${NC}"
    echo -e "${RED}❌ Failed:     ${#FAILED_TESTS[@]}${NC}"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS*100)}")
        echo "Success Rate: ${success_rate}%"
    fi
    
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  Failed tests:${NC}"
        for test_name in "${FAILED_TESTS[@]}"; do
            echo "   - $test_name"
        done
    fi
    
    echo ""
}

# Main execution
main() {
    print_header "URL Shortener Service Validation"
    echo "Testing service at: $BASE_URL"
    echo "Timestamp: $(date -Iseconds)"
    echo ""
    
    # Basic connectivity
    if ! test_health_check; then
        echo ""
        echo -e "${RED}❌ Health check failed. Service may not be running.${NC}"
        echo "   Make sure the service is accessible at $BASE_URL"
        print_summary
        exit 1
    fi
    
    echo ""
    
    # Core functionality tests
    echo "Testing create short URL..."
    test_url="https://example.com/test/$(date +%s)"
    
    if response=$(curl -s --max-time 5 -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"$test_url\"}" 2>/dev/null); then
        
        short_code=$(echo "$response" | grep -o '"short_code":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$short_code" ]; then
            test_result "Create Short URL" 1 "Code: $short_code"
            test_get_url_info "$short_code"
            test_redirect "$short_code"
        else
            test_result "Create Short URL" 0 "No short code in response"
        fi
    else
        test_result "Create Short URL" 0 "Request failed"
    fi
    
    echo ""
    
    # Advanced functionality tests
    echo "Testing custom short code..."
    custom_code="test$(date +%s)"
    
    if response=$(curl -s --max-time 5 -X POST "$BASE_URL/api/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"https://github.com/example/repo\",\"custom_code\":\"$custom_code\"}" 2>/dev/null); then
        
        returned_code=$(echo "$response" | grep -o '"short_code":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$returned_code" = "$custom_code" ]; then
            test_result "Custom Short Code" 1 "Code: $custom_code"
            test_duplicate_code "$custom_code"
        else
            test_result "Custom Short Code" 0 "Code mismatch: got $returned_code"
        fi
    else
        test_result "Custom Short Code" 0 "Request failed"
    fi
    
    test_invalid_url
    test_nonexistent_code
    
    echo ""
    
    # Additional endpoints
    test_stats
    test_web_interface
    
    # Print summary
    print_summary
    
    # Exit with appropriate code
    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [BASE_URL]"
    echo ""
    echo "Validate URL Shortener service functionality"
    echo ""
    echo "Arguments:"
    echo "  BASE_URL    Base URL of the service (default: http://localhost:9200)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 http://localhost:9200"
    echo "  $0 http://example.com"
    exit 0
fi

main

