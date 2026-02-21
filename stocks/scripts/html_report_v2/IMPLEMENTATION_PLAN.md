# HTML Report Generator V2 - Implementation Plan

## Overview
Build a clean, maintainable HTML report generator for options analysis results with proper separation of concerns.

## Core Requirements

### 1. Data Input & Processing
- [ ] **Input Handler**
  - Accept pandas DataFrame with options data
  - Validate required columns (ticker, option_type, strike_price, etc.)
  - Split data into calls/puts based on `option_type` column
  - Handle missing data gracefully

- [ ] **Data Preparation**
  - Normalize column names (handle variations: current_price, curr_price, etc.)
  - Format numeric values (currency, percentages, dates)
  - Format dates (expiration dates, timestamps)
  - Calculate derived values (age in seconds → human-readable)
  - Preserve raw values for filtering/sorting
  - Reorder columns according to display preferences

### 2. HTML Structure
- [ ] **Document Structure**
  - HTML5 doctype
  - Meta tags (charset, viewport)
  - Title based on option type (Calls/Puts/Mixed)
  - Embedded CSS (single file, no external dependencies)
  - Embedded JavaScript (single file, no external dependencies)

- [ ] **Header Section**
  - Title: "Options Analysis Results" (with emoji icon)
  - Tab navigation buttons: Calls, Puts, Comprehensive Analysis
  - Generation timestamp
  - Total results count
  - Desktop/mobile specific instructions

- [ ] **Tab Content Sections**
  - Calls tab (if calls data exists)
  - Puts tab (if puts data exists)
  - Comprehensive Analysis tab (always present)
  - Each tab should be completely independent (no shared state issues)

### 3. Desktop Table View
- [ ] **Table Structure**
  - Sortable columns (click headers to sort)
  - Two-row header system:
    - Row 1: Group headers (e.g., "Strike Price" spanning Short/Long columns)
    - Row 2: Individual column headers with Short/Long labels
  - Column grouping (Short/Long pairs for strike, premium, expiry, etc.)
  - Hidden columns (default-hidden, toggleable)
  - Always-hidden columns (never shown)

- [ ] **Table Features**
  - Column sorting (ascending/descending)
  - Visual sort indicators (↑ ↓ ↕)
  - Row hover effects
  - Alternating row colors for readability
  - Responsive column widths
  - Horizontal scroll for many columns

- [ ] **Data Display**
  - Format numbers (currency, percentages)
  - Format dates (YYYY-MM-DD)
  - Format age (seconds → "5 mins", "2 hrs", "1.2 days")
  - Color coding (positive/negative price changes)
  - Handle missing/null values (show empty or "N/A")

### 4. Mobile Card View
- [ ] **Card Structure**
  - One card per row of data
  - Card header: Ticker, Current Price, Price Change
  - Primary metrics: Strike, Expiry, Net Daily, Daily Premium
  - Expandable details section (hidden by default)
  - "Show More Details" toggle button

- [ ] **Card Features**
  - Tap to expand/collapse details
  - Organized sections: Option Details, Greeks, Premium & Returns, Other
  - Color-coded price changes
  - Responsive card width
  - Proper spacing between cards

- [ ] **Responsive Behavior**
  - Cards shown on mobile (≤768px)
  - Table shown on desktop (>768px)
  - CSS media queries control visibility
  - No JavaScript needed for basic responsive switching

### 5. Filtering System
- [ ] **Filter UI**
  - "Filter" button to show/hide filter section
  - Filter input field with placeholder examples
  - "Add Filter" button
  - "Clear All" button
  - Active filters display (with remove buttons)
  - Filter logic toggle (AND/OR)

- [ ] **Filter Capabilities**
  - Numeric comparisons: `pe_ratio > 20`, `volume < 100`
  - Field existence: `volume exists`, `delta not_exists`
  - Field-to-field comparisons: `num_contracts > volume`
  - Mathematical expressions: `curr_price*1.05 < strike_price`
  - Multiple filters with AND/OR logic
  - Real-time filtering (as you type or on Enter)

- [ ] **Filter Persistence**
  - Save filters in URL hash/query params
  - Restore filters on page load
  - Shareable URLs with filters

### 6. Column Management
- [ ] **Column Visibility**
  - "Show hidden columns" toggle button
  - Default-hidden columns (show when toggled)
  - Always-hidden columns (never shown, used for sorting only)
  - Visual indicator for hidden columns

- [ ] **Column Display**
  - Compact header names (truncate long names)
  - Full column names in filter section
  - Column name normalization for filtering
  - Column grouping labels (Short/Long)

### 7. Comprehensive Analysis Section
- [ ] **Summary Statistics**
  - Spread impact statistics (average, median, distribution)
  - Liquidity distribution (high/medium/low counts)
  - Risk metrics

- [ ] **Detailed Analysis**
  - Top N trades (sorted by score or net daily premium)
  - For each trade:
    - Position structure (option type, strikes, spread width)
    - Option tickers (short and long)
    - Premium breakdown
    - Risk assessment (assignment risk, liquidity, valuation)
    - Recommendation (STRONG BUY, BUY, HOLD, PASS)
    - Score calculation breakdown

### 8. Styling (CSS)
- [ ] **Color Scheme**
  - Primary purple/blue theme
  - White content areas
  - Gray borders and separators
  - Green for positive changes
  - Red for negative changes
  - Yellow/orange for warnings

- [ ] **Typography**
  - Clean, readable fonts
  - Appropriate font sizes (responsive)
  - Proper line heights
  - Text truncation where needed

- [ ] **Layout**
  - Centered container with max-width
  - Proper padding and margins
  - Responsive breakpoints (768px mobile/desktop)
  - Flexbox/Grid for layouts
  - Sticky header (optional)

- [ ] **Components**
  - Button styles (primary, secondary, clear)
  - Input field styles
  - Table styles (borders, hover, sort indicators)
  - Card styles (shadows, borders, hover)
  - Tab button styles (active/inactive states)
  - Filter section styles
  - Badge/risk indicator styles

### 9. Interactivity (JavaScript)
- [ ] **Tab Switching**
  - Click tab button → switch to that tab
  - Hide all tabs, show selected tab
  - Update active button styling
  - Preserve filter state per tab (optional)

- [ ] **Table Sorting**
  - Click column header → sort by that column
  - Toggle ascending/descending
  - Handle numeric, text, and date sorting
  - Update sort indicators
  - Preserve sort state

- [ ] **Filtering**
  - Parse filter expressions
  - Validate filter syntax
  - Apply filters to table rows
  - Apply filters to cards
  - Update visible row counts
  - Show filter errors

- [ ] **Card Expansion**
  - Toggle card details visibility
  - Smooth expand/collapse animation
  - Update button text/icon

- [ ] **Column Visibility**
  - Toggle hidden columns
  - Update table display
  - Persist preference (localStorage, optional)

- [ ] **URL State Management**
  - Read filters from URL on load
  - Update URL when filters change
  - Handle browser back/forward buttons

- [ ] **Utilities**
  - Extract numeric values from formatted strings
  - Parse date strings
  - Normalize column names
  - Debounce filter input (optional)

### 10. Data Formatting Utilities
- [ ] **Number Formatting**
  - Currency: `$1,234.56`
  - Percentages: `12.34%`
  - Large numbers: `1.23M`, `1.23B`
  - Decimals: appropriate precision

- [ ] **Date Formatting**
  - Expiration dates: `YYYY-MM-DD`
  - Timestamps: `YYYY-MM-DD HH:MM:SS`
  - Age formatting: `5 secs`, `2 mins`, `3 hrs`, `1.2 days`

- [ ] **Text Formatting**
  - Truncate long headers
  - Escape HTML special characters
  - Normalize column names (lowercase, underscores)

### 11. Error Handling
- [ ] **Data Validation**
  - Check for required columns
  - Handle missing data gracefully
  - Validate data types
  - Show error messages for invalid data

- [ ] **Filter Errors**
  - Validate filter syntax
  - Show helpful error messages
  - Suggest corrections

- [ ] **JavaScript Errors**
  - Try-catch blocks for critical operations
  - Console logging for debugging
  - Graceful degradation

### 12. Performance
- [ ] **Optimization**
  - Efficient DOM manipulation
  - Debounce filter input
  - Virtual scrolling for large datasets (optional)
  - Lazy loading of details (optional)

### 13. Testing & Validation
- [ ] **Test Cases**
  - Empty DataFrame
  - DataFrame with only calls
  - DataFrame with only puts
  - DataFrame with both calls and puts
  - Missing columns
  - Invalid data types
  - Large datasets (1000+ rows)

- [ ] **Browser Testing**
  - Chrome/Edge
  - Firefox
  - Safari
  - Mobile browsers

## File Structure

```
scripts/html_report_v2/
├── __init__.py                 # Package init, export main function
├── config.py                   # Configuration constants
├── data_processor.py           # Data preparation and formatting
├── html_builder.py             # HTML structure generation
├── table_builder.py            # Desktop table HTML generation
├── card_builder.py             # Mobile card HTML generation
├── analysis_builder.py         # Comprehensive analysis HTML
├── styles.py                   # CSS generation
├── scripts.py                  # JavaScript generation
└── generator.py                # Main orchestrator
```

## Implementation Order

1. **Phase 1: Core Structure**
   - Set up file structure
   - Create config with constants
   - Build basic HTML skeleton
   - Implement data processor

2. **Phase 2: Desktop Table**
   - Build table HTML structure
   - Implement column grouping
   - Add sorting functionality
   - Style the table

3. **Phase 3: Mobile Cards**
   - Build card HTML structure
   - Implement expand/collapse
   - Style the cards
   - Test responsive behavior

4. **Phase 4: Filtering**
   - Build filter UI
   - Implement filter parsing
   - Add filter application logic
   - Add URL state management

5. **Phase 5: Analysis Section**
   - Build summary statistics
   - Build detailed analysis
   - Style analysis sections

6. **Phase 6: Polish**
   - Add error handling
   - Optimize performance
   - Test edge cases
   - Document code

## Key Design Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **No Shared State**: Tabs are completely independent
3. **CSS-First Responsive**: Use CSS media queries, not JavaScript
4. **Progressive Enhancement**: Basic functionality works without JS
5. **Maintainability**: Clear code structure, good naming, comments
6. **Testability**: Functions are pure where possible, easy to test

## Critical Fixes from V1

1. **Tab Independence**: Each tab's HTML is completely separate, no shared elements
2. **Stats Removal**: No stats section after cards (removed completely)
3. **Desktop Table Display**: CSS ensures tables show on desktop for all tabs
4. **Mobile Card Display**: CSS ensures cards show on mobile for all tabs
5. **Proper Tab Switching**: JavaScript only toggles classes, CSS handles display

