/**
 * URL Shortener Client-Side JavaScript
 */

// Form validation and enhancement
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('shortenForm');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            const urlInput = document.getElementById('url');
            const customCodeInput = document.getElementById('custom_code');
            
            // Validate URL
            if (!urlInput.value.match(/^https?:\/\/.+/)) {
                e.preventDefault();
                alert('Please enter a valid URL starting with http:// or https://');
                urlInput.focus();
                return false;
            }
            
            // Validate custom code if provided
            if (customCodeInput.value.trim()) {
                const code = customCodeInput.value.trim();
                if (!code.match(/^[a-zA-Z0-9_-]{4,20}$/)) {
                    e.preventDefault();
                    alert('Custom code must be 4-20 characters (letters, numbers, hyphens, underscores only)');
                    customCodeInput.focus();
                    return false;
                }
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Shortening...';
        });
        
        // Real-time validation for custom code
        const customCodeInput = document.getElementById('custom_code');
        if (customCodeInput) {
            customCodeInput.addEventListener('input', function() {
                const value = this.value.trim();
                if (value && !value.match(/^[a-zA-Z0-9_-]*$/)) {
                    this.setCustomValidity('Only letters, numbers, hyphens, and underscores allowed');
                } else {
                    this.setCustomValidity('');
                }
            });
        }
    }
});

// Copy to clipboard functionality (used in result.html)
function copyToClipboard() {
    const input = document.getElementById('shortUrl');
    const btn = document.getElementById('copyBtn');
    
    if (!input || !btn) return;
    
    input.select();
    input.setSelectionRange(0, 99999); // For mobile devices
    
    navigator.clipboard.writeText(input.value).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = '✅ Copied!';
        btn.classList.add('copied');
        
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        // Fallback for older browsers
        try {
            document.execCommand('copy');
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Copied!';
            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 2000);
        } catch (e) {
            alert('Failed to copy. Please copy manually.');
        }
    });
}

// Optional: Add API integration for AJAX submission
function shortenUrlAjax(url, customCode) {
    return fetch('/api/shorten', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            url: url,
            custom_code: customCode || null
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.detail || 'Failed to shorten URL');
            });
        }
        return response.json();
    });
}






