"""
JavaScript code generator for HTML report.
"""


def get_javascript():
    """Get JavaScript code for table sorting and filtering functionality.
    
    Returns:
        String containing JavaScript code
    """
    return """        // Namespaced state for calls and puts
        let sortDirection = {};
        let currentSortColumn = {};
        let activeFilters = {};
        let filterLogic = {};
        
        // Initialize state for a prefix
        function initStateForPrefix(prefix) {
            if (!sortDirection[prefix]) sortDirection[prefix] = {};
            if (currentSortColumn[prefix] === undefined) currentSortColumn[prefix] = -1;
            if (!activeFilters[prefix]) activeFilters[prefix] = [];
            if (!filterLogic[prefix]) filterLogic[prefix] = 'AND';
        }
        
        // Column name mapping (original names to display names) - namespaced by table
        const columnMap = {};
        
        // Initialize column map for a specific table
        function initColumnMap(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            // Return early if table doesn't exist (prevents null reference errors)
            if (!table) {
                return;
            }
            // Only use column headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            headers.forEach((th, index) => {
                const colName = th.textContent.trim().replace(/\\s+/g, ' ').replace(/\\n/g, '');
                columnMap[colName] = index;
                // Also map common variations
                const lowerName = colName.toLowerCase();
                if (lowerName.includes('pe') && lowerName.includes('ratio')) columnMap['pe_ratio'] = index;
                if (lowerName.includes('market') && lowerName.includes('cap')) columnMap['market_cap_b'] = index;
                if (lowerName.includes('current') || lowerName.includes('curr')) columnMap['current_price'] = index;
                if (lowerName.includes('strike')) columnMap['strike_price'] = index;
                if (lowerName.includes('premium') && !lowerName.includes('daily') && !lowerName.includes('total')) columnMap['option_premium'] = index;
                if (lowerName.includes('daily') && lowerName.includes('prem')) columnMap['daily_premium'] = index;
                if (lowerName.includes('volume')) columnMap['volume'] = index;
                if (lowerName.includes('delta')) columnMap['delta'] = index;
                if (lowerName.includes('theta')) columnMap['theta'] = index;
                if (lowerName.includes('days') && lowerName.includes('expir')) columnMap['days_to_expiry'] = index;
                if (lowerName.includes('net') && lowerName.includes('daily')) columnMap['net_daily_premium'] = index;
                if (lowerName.includes('net') && lowerName.includes('prem') && !lowerName.includes('daily')) columnMap['net_premium'] = index;
            });
        }
        
        // Get raw numeric value from a cell
        function getRawValue(cell) {
            const rawValue = cell.getAttribute('data-raw');
            if (rawValue !== null && rawValue !== '') {
                const num = parseFloat(rawValue);
                if (!isNaN(num)) return num;
            }
            // Fallback: try to parse from text
            const text = cell.textContent.trim();
            const num = parseFloat(text.replace(/[^0-9.-]/g, ''));
            return isNaN(num) ? null : num;
        }
        
        // Get raw text value from a cell
        function getRawText(cell) {
            const rawValue = cell.getAttribute('data-raw-text');
            if (rawValue !== null) return rawValue;
            return cell.textContent.trim();
        }
        
        // Evaluate a mathematical expression like "curr_price*1.05" or "strike_price+100"
        function evaluateMathExpression(expression, row, tableId) {
            // Extract field names from expression (alphanumeric + underscore)
            const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
            const fields = expression.match(fieldPattern) || [];
            
            // Filter out JavaScript keywords and common math functions
            const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
            const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
            
            // Get unique fields and their values
            const fieldValues = {};
            for (const field of validFields) {
                // Skip if we already processed this field
                if (fieldValues.hasOwnProperty(field)) continue;
                
                const colIndex = findColumnIndex(field, tableId);
                if (colIndex >= 0) {
                    const cell = row.cells[colIndex];
                    if (cell) {
                        const value = getRawValue(cell);
                        if (value !== null) {
                            fieldValues[field] = value;
                        } else {
                            console.warn('Field', field, 'found but has no numeric value');
                            return null; // Field not found or has no value
                        }
                    } else {
                        console.warn('Field', field, 'column index', colIndex, 'but cell not found');
                        return null; // Cell not found
                    }
                } else {
                    console.warn('Field', field, 'not found in table columns');
                    return null; // Field not found
                }
            }
            
            // Replace field names with their values (in reverse order of length to avoid partial replacements)
            let evalExpr = expression;
            const sortedFields = Object.keys(fieldValues).sort((a, b) => b.length - a.length);
            for (const field of sortedFields) {
                const value = fieldValues[field];
                // Replace field name, ensuring we match whole identifiers (not part of another identifier)
                // Escape special regex characters in the field name
                const escapedField = field.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
                // Match field name as whole word (not part of another identifier)
                const regex = new RegExp('(^|[^a-zA-Z0-9_])' + escapedField + '([^a-zA-Z0-9_]|$)', 'g');
                evalExpr = evalExpr.replace(regex, function(match, before, after) {
                    // Preserve the before/after characters, replace only the field name
                    return before + value + after;
                });
            }
            
            // Safely evaluate the expression
            try {
                // Use Function constructor for safer evaluation
                // This evaluates basic math expressions with +, -, *, /
                const result = Function('"use strict"; return (' + evalExpr + ')')();
                return typeof result === 'number' && !isNaN(result) ? result : null;
            } catch (e) {
                console.error('Error evaluating math expression:', evalExpr, e);
                return null;
            }
        }
        
        // Get all available column names for error messages
        function getAllColumnNames(tableId) {
            const table = document.getElementById(tableId);
            // Return empty array if table doesn't exist (prevents null reference errors)
            if (!table) {
                return [];
            }
            // Only get headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : [];
            const columnNames = [];
            headers.forEach(th => {
                const filterableName = th.getAttribute('data-filterable-name');
                if (filterableName) {
                    columnNames.push(filterableName);
                } else {
                    // Get text from the filterable name span if available, otherwise use display name
                    const filterableSpan = th.querySelector('.column-name-filterable');
                    const displaySpan = th.querySelector('.column-name-display');
                    if (filterableSpan && filterableSpan.textContent) {
                        columnNames.push(filterableSpan.textContent.trim());
                    } else if (displaySpan && displaySpan.textContent) {
                        columnNames.push(displaySpan.textContent.trim().replace(/\\s+/g, ' '));
                    } else {
                        const headerText = th.textContent.trim().replace(/\\s+/g, ' ').replace(/\\n/g, '');
                        if (headerText) {
                            columnNames.push(headerText);
                        }
                    }
                }
            });
            return columnNames;
        }
        
        // Find column index by field name (supports substring matching)
        function findColumnIndex(fieldName, tableId) {
            const table = document.getElementById(tableId);
            // Return -1 if table doesn't exist (prevents null reference errors)
            if (!table) {
                return -1;
            }
            // Only get headers from the column-header-row, not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : [];
            const lowerField = fieldName.toLowerCase().replace(/\\s+/g, '_').trim();
            
            if (!lowerField) return -1;
            
            // First, try to match against data-filterable-name attribute (most reliable)
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Exact match (most reliable)
                    if (normalizedFilterable === lowerField) {
                        return i;
                    }
                }
            }
            
            // Try exact match against header text (normalized)
            for (let i = 0; i < headers.length; i++) {
                const headerText = headers[i].textContent.trim().toLowerCase().replace(/\\s+/g, '_').replace(/\\n/g, '').trim();
                if (headerText === lowerField) {
                    return i;
                }
            }
            
            // Try matching against data-filterable-name with normalization
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Check if field name is contained in filterable name or vice versa
                    // e.g., 'curr_price' should match 'current_price'
                    if (normalizedFilterable === lowerField || 
                        normalizedFilterable.replace('current', 'curr') === lowerField ||
                        lowerField.replace('curr', 'current') === normalizedFilterable) {
                        return i;
                    }
                }
            }
            
            // Try substring match on filterable name (but prefer exact matches)
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\\s+/g, '_').trim();
                    // Only match if one contains the other (but not too loose)
                    if ((normalizedFilterable.length >= 3 && lowerField.length >= 3) &&
                        (normalizedFilterable.includes(lowerField) || lowerField.includes(normalizedFilterable))) {
                        return i;
                    }
                }
            }
            
            // Try column map
            if (columnMap[fieldName] !== undefined) {
                return columnMap[fieldName];
            }
            
            return -1;
        }
        
        // Parse market cap value with B/M/T suffixes
        function parseMarketCapValue(valueStr) {
            valueStr = valueStr.trim().toUpperCase();
            if (valueStr.endsWith('T')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e12;
            } else if (valueStr.endsWith('B')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e9;
            } else if (valueStr.endsWith('M')) {
                return parseFloat(valueStr.slice(0, -1)) * 1e6;
            }
            return parseFloat(valueStr);
        }
        
        // Parse filter expression
        function parseFilterExpression(expression, tableId) {
            expression = expression.trim();
            if (!expression) return null;
            
            // Handle exists/not_exists
            const existsMatch = expression.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\\s+(exists|not_exists)$/i);
            if (existsMatch) {
                return {
                    field: existsMatch[1],
                    operator: existsMatch[2].toLowerCase(),
                    value: null,
                    isFieldComparison: false
                };
            }
            
            // Parse comparison operators
            const operators = ['>=', '<=', '==', '!=', '>', '<'];
            for (const op of operators) {
                if (expression.includes(op)) {
                    const parts = expression.split(op, 2);
                    if (parts.length === 2) {
                        const fieldExpr = parts[0].trim();
                        const valueStr = parts[1].trim();
                        
                        // Check for mathematical expressions in field FIRST
                        const hasMath = /[+\\-*/]/.test(fieldExpr);
                        if (hasMath) {
                            // Check if the value is also a field (field-to-field comparison with math)
                            // Only check if tableId is provided
                            if (tableId) {
                                const valueColIndex = findColumnIndex(valueStr, tableId);
                                if (valueColIndex >= 0) {
                                    // Math expression compared to another field: "curr_price*1.05 < strike_price"
                                    return {
                                        field: fieldExpr,
                                        operator: op,
                                        value: valueStr,
                                        isFieldComparison: true,
                                        hasMath: true
                                    };
                                }
                            }
                            // Math expression compared to a value: "curr_price*1.05 < 150"
                            let value = valueStr;
                            const numValue = parseFloat(valueStr);
                            if (!isNaN(numValue)) {
                                value = numValue;
                            }
                            return {
                                field: fieldExpr,
                                operator: op,
                                value: value,
                                isFieldComparison: false,
                                hasMath: true
                            };
                        }
                        
                        // Check if value is a field name (field-to-field comparison without math)
                        // Only check if tableId is provided
                        if (tableId) {
                            const valueColIndex = findColumnIndex(valueStr, tableId);
                            if (valueColIndex >= 0) {
                                return {
                                    field: fieldExpr,
                                    operator: op,
                                    value: valueStr,
                                    isFieldComparison: true
                                };
                            }
                        }
                        
                        // Regular value comparison
                        let value = valueStr;
                        // Try to parse as number
                        if (fieldExpr.toLowerCase().includes('market_cap') || fieldExpr.toLowerCase().includes('market cap')) {
                            value = parseMarketCapValue(valueStr);
                        } else {
                            const numValue = parseFloat(valueStr);
                            if (!isNaN(numValue)) {
                                value = numValue;
                            }
                        }
                        
                        return {
                            field: fieldExpr,
                            operator: op,
                            value: value,
                            isFieldComparison: false
                        };
                    }
                }
            }
            
            return null;
        }
        
        // Evaluate filter expression for a row
        function evaluateFilter(filter, row, tableId) {
            // Handle exists/not_exists (only works for single fields, not expressions)
            if (filter.operator === 'exists' || filter.operator === 'not_exists') {
                if (filter.hasMath) return false; // Can't use exists/not_exists with math expressions
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    console.warn('Column not found for filter:', filter.field);
                    return false;
                }
                const cell = row.cells[colIndex];
                if (!cell) return false;
                
                if (filter.operator === 'exists') {
                    const rawValue = cell.getAttribute('data-raw');
                    const text = cell.textContent.trim();
                    return (rawValue !== null && rawValue !== '') || (text !== '' && text !== 'N/A');
                }
                if (filter.operator === 'not_exists') {
                    const rawValue = cell.getAttribute('data-raw');
                    const text = cell.textContent.trim();
                    return (rawValue === null || rawValue === '') && (text === '' || text === 'N/A');
                }
            }
            
            // Handle mathematical expressions
            if (filter.hasMath) {
                // Evaluate the mathematical expression on the left side
                const leftValue = evaluateMathExpression(filter.field, row, tableId);
                if (leftValue === null) {
                    console.warn('Could not evaluate math expression:', filter.field);
                    return false; // Can't evaluate expression
                }
                
                // Get the right side value
                let rightValue = null;
                if (filter.isFieldComparison) {
                    // Right side is also a field: "curr_price*1.05 < strike_price"
                    const valueColIndex = findColumnIndex(filter.value, tableId);
                    if (valueColIndex < 0) {
                        console.warn('Column not found for math expression comparison value:', filter.value);
                        return false;
                    }
                    const valueCell = row.cells[valueColIndex];
                    if (!valueCell) return false;
                    rightValue = getRawValue(valueCell);
                } else {
                    // Right side is a numeric value: "curr_price*1.05 < 150"
                    rightValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
                }
                
                if (rightValue === null || isNaN(rightValue)) {
                    console.warn('Invalid right side value for math expression:', filter.value);
                    return false;
                }
                
                // Compare the evaluated expression with the right side
                switch (filter.operator) {
                    case '>': return leftValue > rightValue;
                    case '>=': return leftValue >= rightValue;
                    case '<': return leftValue < rightValue;
                    case '<=': return leftValue <= rightValue;
                    case '==': return Math.abs(leftValue - rightValue) < 0.0001;
                    case '!=': return Math.abs(leftValue - rightValue) >= 0.0001;
                    default: return false;
                }
            }
            
            // Handle field-to-field comparison (without math)
            if (filter.isFieldComparison) {
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    console.warn('Column not found for filter field:', filter.field);
                    return false;
                }
                const cell = row.cells[colIndex];
                if (!cell) return false;
                
                const valueColIndex = findColumnIndex(filter.value, tableId);
                if (valueColIndex < 0) {
                    console.warn('Column not found for filter comparison value:', filter.value);
                    return false;
                }
                const valueCell = row.cells[valueColIndex];
                if (!valueCell) return false;
                
                const cellValue = getRawValue(cell);
                const compareValue = getRawValue(valueCell);
                if (cellValue === null || compareValue === null) return false;
                
                switch (filter.operator) {
                    case '>': return cellValue > compareValue;
                    case '>=': return cellValue >= compareValue;
                    case '<': return cellValue < compareValue;
                    case '<=': return cellValue <= compareValue;
                    case '==': return Math.abs(cellValue - compareValue) < 0.0001;
                    case '!=': return Math.abs(cellValue - compareValue) >= 0.0001;
                    default: return false;
                }
            }
            
            // Handle value comparison (regular field to value)
            const colIndex = findColumnIndex(filter.field, tableId);
            if (colIndex < 0) {
                console.warn('Column not found for filter field:', filter.field);
                return false;
            }
            const cell = row.cells[colIndex];
            if (!cell) return false;
            
            const cellValue = getRawValue(cell);
            if (cellValue === null) return false;
            
            const filterValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
            if (isNaN(filterValue)) {
                // String comparison
                const cellText = getRawText(cell).toLowerCase();
                const filterText = String(filter.value).toLowerCase();
                switch (filter.operator) {
                    case '==': return cellText === filterText;
                    case '!=': return cellText !== filterText;
                    default: return false;
                }
            }
            
            // Numeric comparison
            switch (filter.operator) {
                case '>': return cellValue > filterValue;
                case '>=': return cellValue >= filterValue;
                case '<': return cellValue < filterValue;
                case '<=': return cellValue <= filterValue;
                case '==': return Math.abs(cellValue - filterValue) < 0.0001;
                case '!=': return Math.abs(cellValue - filterValue) >= 0.0001;
                default: return false;
            }
        }
        
        // Apply all filters to table
        function applyFilters(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const errorDiv = document.getElementById(prefix + 'filterError');
            
            if (errorDiv) errorDiv.textContent = '';
            
            initStateForPrefix(prefix);
            
            if (activeFilters[prefix].length === 0) {
                rows.forEach(row => {
                    row.style.display = '';
                });
                updateVisibleCount(prefix);
                applyRowStriping(prefix);
                syncCardVisibility(prefix);
                return;
            }
            
            rows.forEach(row => {
                let matches = activeFilters[prefix].map(filter => evaluateFilter(filter, row, tableId));
                
                let shouldShow;
                if (filterLogic[prefix] === 'AND') {
                    shouldShow = matches.every(m => m);
                } else {
                    shouldShow = matches.some(m => m);
                }
                
                row.style.display = shouldShow ? '' : 'none';
            });
            
            updateVisibleCount(prefix);
            applyRowStriping(prefix);
            syncCardVisibility(prefix);
        }
        
        // Validate filter fields exist
        function validateFilter(filter, tableId) {
            const errorMessages = [];
            
            // Check main field
            if (filter.field && !filter.hasMath) {
                // For non-math expressions, check if field exists
                const colIndex = findColumnIndex(filter.field, tableId);
                if (colIndex < 0) {
                    const availableColumns = getAllColumnNames(tableId);
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(filter.field.toLowerCase()) || 
                        filter.field.toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.field}" not found. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}? `;
                    }
                    // Show all available columns (not just first 10) so user can see what's available
                    errorMsg += `Available columns: ${availableColumns.join(', ')}`;
                    errorMessages.push(errorMsg);
                }
            } else if (filter.hasMath) {
                // For math expressions, extract field names and validate them
                const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
                const fields = filter.field.match(fieldPattern) || [];
                const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
                const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
                
                for (const field of validFields) {
                    const colIndex = findColumnIndex(field, tableId);
                    if (colIndex < 0) {
                        errorMessages.push(`Field "${field}" in expression "${filter.field}" not found.`);
                    }
                }
            }
            
            // Check comparison field if it's a field-to-field comparison
            if (filter.isFieldComparison && filter.value) {
                const valueColIndex = findColumnIndex(filter.value, tableId);
                if (valueColIndex < 0) {
                    const availableColumns = getAllColumnNames(tableId);
                    const suggestions = availableColumns.filter(col => 
                        col.toLowerCase().includes(String(filter.value).toLowerCase()) || 
                        String(filter.value).toLowerCase().includes(col.toLowerCase())
                    ).slice(0, 5);
                    
                    let errorMsg = `Column "${filter.value}" not found for comparison. `;
                    if (suggestions.length > 0) {
                        errorMsg += `Did you mean: ${suggestions.join(', ')}? `;
                    }
                    // Show all available columns so user can see what's available
                    errorMsg += `Available columns: ${availableColumns.join(', ')}`;
                    errorMessages.push(errorMsg);
                }
            }
            
            return errorMessages;
        }
        
        // Add filter
        function addFilter(prefix) {
            const input = document.getElementById(prefix + 'filterInput');
            const expression = input.value.trim();
            const errorDiv = document.getElementById(prefix + 'filterError');
            const tableId = prefix + 'resultsTable';
            
            if (errorDiv) errorDiv.textContent = '';
            
            if (!expression) return;
            
            initStateForPrefix(prefix);
            
            const filter = parseFilterExpression(expression, tableId);
            if (!filter) {
                if (errorDiv) {
                    errorDiv.textContent = 'Invalid filter expression. Format: field operator value (e.g., "pe_ratio > 20", "volume exists")';
                }
                return;
            }
            
            // Validate filter fields exist
            const validationErrors = validateFilter(filter, tableId);
            if (validationErrors.length > 0) {
                if (errorDiv) {
                    errorDiv.textContent = validationErrors.join(' ');
                    errorDiv.style.display = 'block';
                    errorDiv.style.color = '#dc3545';
                    errorDiv.style.backgroundColor = '#f8d7da';
                }
                console.error('Filter validation errors:', validationErrors);
                return;
            }
            
            // Clear any previous errors if validation passes
            if (errorDiv) {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }
            
            activeFilters[prefix].push(filter);
            input.value = '';
            
            // Update filter display
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Remove filter
        function removeFilter(prefix, index) {
            initStateForPrefix(prefix);
            activeFilters[prefix].splice(index, 1);
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Clear all filters
        function clearFilters(prefix) {
            initStateForPrefix(prefix);
            activeFilters[prefix] = [];
            updateFilterDisplay(prefix);
            applyFilters(prefix);
            updateURL();
        }
        
        // Update filter display
        function updateFilterDisplay(prefix) {
            const container = document.getElementById(prefix + 'activeFilters');
            if (!container) return;
            
            initStateForPrefix(prefix);
            container.innerHTML = '';
            
            if (activeFilters[prefix].length === 0) {
                container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No active filters</p>';
                return;
            }
            
            activeFilters[prefix].forEach((filter, index) => {
                const filterDiv = document.createElement('div');
                filterDiv.style.cssText = 'display: inline-block; margin: 5px; padding: 5px 10px; background: #667eea; color: white; border-radius: 5px; font-size: 0.9em;';
                
                const filterText = document.createElement('span');
                filterText.textContent = `${filter.field} ${filter.operator} ${filter.value !== null ? filter.value : ''}`;
                filterDiv.appendChild(filterText);
                
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '×';
                removeBtn.style.cssText = 'margin-left: 8px; background: rgba(255,255,255,0.3); border: none; color: white; cursor: pointer; border-radius: 3px; padding: 2px 6px;';
                removeBtn.onclick = () => removeFilter(prefix, index);
                filterDiv.appendChild(removeBtn);
                
                container.appendChild(filterDiv);
            });
        }
        
        // Update filter logic
        function updateFilterLogic(prefix, logic) {
            initStateForPrefix(prefix);
            filterLogic[prefix] = logic;
            applyFilters(prefix);
            updateURL();
        }
        
        // Update visible count
        function updateVisibleCount(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const visibleRows = Array.from(tbody.querySelectorAll('tr')).filter(row => row.style.display !== 'none');
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleRows.length;
            }
        }
        
        // Apply row striping based on visible rows
        function applyRowStriping(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Remove all even-row classes first
            rows.forEach(row => row.classList.remove('even-row'));
            
            // Apply even-row class to every other visible row
            let visibleIndex = 0;
            rows.forEach(row => {
                if (row.style.display !== 'none') {
                    if (visibleIndex % 2 === 1) {
                        row.classList.add('even-row');
                    }
                    visibleIndex++;
                }
            });
        }
        
        // Apply column striping based on visible columns
        function applyColumnStriping(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const thead = table.querySelector('thead');
            const tbody = table.querySelector('tbody');
            const wrapperId = prefix + 'tableWrapper';
            const wrapper = document.getElementById(wrapperId);
            
            if (!thead || !tbody) return;
            
            // Only use column headers from the column-header-row, not group headers
            const headerRow = thead.querySelector('tr.column-header-row');
            const headers = headerRow ? Array.from(headerRow.querySelectorAll('th')) : Array.from(thead.querySelectorAll('th'));
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Check if hidden columns are currently hidden
            const hideHidden = wrapper && wrapper.classList.contains('hide-hidden');
            
            // Remove all even-col classes first
            headers.forEach(th => th.classList.remove('even-col'));
            rows.forEach(row => {
                Array.from(row.querySelectorAll('td')).forEach(td => td.classList.remove('even-col'));
            });
            
            // Find visible column indices (columns that are actually displayed)
            const visibleColumnIndices = [];
            headers.forEach((th, index) => {
                // Column is visible if:
                // 1. It doesn't have always-hidden class (always-hidden columns are never visible)
                // 2. It doesn't have is-hidden-col class, OR
                // 3. It has is-hidden-col but hide-hidden is false (showing all columns)
                const isAlwaysHidden = th.classList.contains('always-hidden');
                const isHiddenCol = th.classList.contains('is-hidden-col');
                if (!isAlwaysHidden && (!isHiddenCol || !hideHidden)) {
                    visibleColumnIndices.push(index);
                }
            });
            
            // Apply even-col class to every other visible column
            visibleColumnIndices.forEach((colIndex, visibleIndex) => {
                if (visibleIndex % 2 === 1) {
                    // Add to header
                    if (headers[colIndex]) {
                        headers[colIndex].classList.add('even-col');
                    }
                    // Add to all cells in this column
                    rows.forEach(row => {
                        const cell = row.cells[colIndex];
                        if (cell) {
                            cell.classList.add('even-col');
                        }
                    });
                }
            });
        }
        
        function sortTable(prefix, columnIndex) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr:not([style*="display: none"])'));
            // Only select headers from the column-header-row (second row), not group headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            
            initStateForPrefix(prefix);
            
            // Remove sort classes from all column headers
            headers.forEach(h => {
                h.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Determine sort direction
            if (currentSortColumn[prefix] === columnIndex) {
                sortDirection[prefix][columnIndex] = sortDirection[prefix][columnIndex] === 'asc' ? 'desc' : 'asc';
            } else {
                if (!sortDirection[prefix]) sortDirection[prefix] = {};
                sortDirection[prefix][columnIndex] = 'asc';
                currentSortColumn[prefix] = columnIndex;
            }
            
            // Add sort class to current header
            if (headers[columnIndex]) {
                headers[columnIndex].classList.add(sortDirection[prefix][columnIndex] === 'asc' ? 'sort-asc' : 'sort-desc');
            }
            
            // Check if sorting by change_pct (renamed from price_with_change)
            // We can sort directly using the data-raw attribute which contains the percentage value
            // No need to find a separate column - the change_pct column itself has the percentage in data-raw
            const currentHeader = headers[columnIndex];
            const filterableName = currentHeader.getAttribute('data-filterable-name');
            let sortColumnIndex = columnIndex;
            // For change_pct column, we'll use its own data-raw attribute which contains the percentage
            // The JavaScript sorting code below will automatically use data-raw if available
            
            // Check if this is a date column
            // Note: latest_option_writets is NOT a date column - it's age in seconds (numeric)
            const filterableLower = filterableName ? filterableName.toLowerCase() : '';
            const isDateColumn = filterableLower && (
                filterableLower.includes('expiration_date') ||
                filterableLower.includes('exp_date') ||
                (filterableLower.endsWith('_expiration_date') && !filterableLower.includes('latest_option_writets') && !filterableLower.includes('latest_opt_ts')) ||
                (filterableLower.endsWith('_exp_date') && !filterableLower.includes('latest_option_writets') && !filterableLower.includes('latest_opt_ts')) ||
                filterableLower === 'l_expiration_date' ||
                filterableLower === 'long_expiration_date' ||
                filterableLower === 'expiration_date'
            );
            
            // Check if this is an age column (latest_option_writets - age in seconds)
            const isAgeColumn = filterableLower && (
                filterableLower.includes('latest_option_writets') ||
                filterableLower.includes('latest_opt_ts') ||
                filterableLower.endsWith('_writets')
            );
            
            // Sort rows
            rows.sort((a, b) => {
                const aCell = a.cells[sortColumnIndex];
                const bCell = b.cells[sortColumnIndex];
                if (!aCell || !bCell) return 0;
                
                const aRaw = aCell.getAttribute('data-raw');
                const bRaw = bCell.getAttribute('data-raw');
                
                let comparison = 0;
                
                // Handle age columns (latest_option_writets) - sort as numeric (smaller = more recent)
                if (isAgeColumn) {
                    // Age in seconds - smaller value = more recent
                    let aNum = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bNum = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    if (isNaN(aNum)) {
                        aNum = parseFloat(aCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    if (isNaN(bNum)) {
                        bNum = parseFloat(bCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        comparison = aNum - bNum;  // Smaller age = more recent
                    } else if (isNaN(aNum) && !isNaN(bNum)) {
                        comparison = 1; // a comes after b
                    } else if (!isNaN(aNum) && isNaN(bNum)) {
                        comparison = -1; // a comes before b
                    } else {
                        // Both are NaN, use text comparison
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                }
                // Handle date columns specially
                else if (isDateColumn) {
                    // For date columns, data-raw contains timestamp in milliseconds
                    let aTimestamp = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bTimestamp = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    // If timestamp parsing fails, try to parse from text (YYYY-MM-DD format)
                    if (isNaN(aTimestamp)) {
                        const aText = aCell.textContent.trim();
                        const aDate = new Date(aText);
                        aTimestamp = isNaN(aDate.getTime()) ? NaN : aDate.getTime();
                    }
                    if (isNaN(bTimestamp)) {
                        const bText = bCell.textContent.trim();
                        const bDate = new Date(bText);
                        bTimestamp = isNaN(bDate.getTime()) ? NaN : bDate.getTime();
                    }
                    
                    if (!isNaN(aTimestamp) && !isNaN(bTimestamp)) {
                        comparison = aTimestamp - bTimestamp;
                    } else if (isNaN(aTimestamp) && !isNaN(bTimestamp)) {
                        comparison = 1; // a comes after b
                    } else if (!isNaN(aTimestamp) && isNaN(bTimestamp)) {
                        comparison = -1; // a comes before b
                    } else {
                        // Both are NaN, use text comparison
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                } else {
                    // Regular numeric/text sorting
                    let aNum = aRaw !== null ? parseFloat(aRaw) : NaN;
                    let bNum = bRaw !== null ? parseFloat(bRaw) : NaN;
                    
                    if (isNaN(aNum)) {
                        aNum = parseFloat(aCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    if (isNaN(bNum)) {
                        bNum = parseFloat(bCell.textContent.trim().replace(/[^0-9.-]/g, ''));
                    }
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        comparison = aNum - bNum;
                    } else {
                        const aText = aCell.textContent.trim();
                        const bText = bCell.textContent.trim();
                        comparison = aText.localeCompare(bText);
                    }
                }
                
                return sortDirection[prefix][columnIndex] === 'asc' ? comparison : -comparison;
            });
            
            // Re-append sorted rows (only visible ones)
            const hiddenRows = Array.from(tbody.querySelectorAll('tr[style*="display: none"]'));
            rows.forEach(row => tbody.appendChild(row));
            hiddenRows.forEach(row => tbody.appendChild(row));
            
            // Sync card order with table order
            syncCardOrder(prefix);
            
            updateVisibleCount(prefix);
            applyRowStriping(prefix);
        }
        
        // Sync card order with table row order
        function syncCardOrder(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            const tbody = table ? table.querySelector('tbody') : null;
            const cardWrapperId = prefix + 'cardWrapper';
            const cardWrapper = document.getElementById(cardWrapperId);
            if (!tbody || !cardWrapper) return;
            
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const cards = cardWrapper ? Array.from(cardWrapper.querySelectorAll('.data-card')) : [];
            
            // Create a map of row index to card
            const cardMap = new Map();
            cards.forEach(card => {
                const rowIndex = parseInt(card.getAttribute('data-row-index'));
                if (!isNaN(rowIndex)) {
                    cardMap.set(rowIndex, card);
                }
            });
            
            // Reorder cards to match table row order
            rows.forEach((row, index) => {
                const rowIndex = parseInt(row.getAttribute('data-row-index') || index);
                const card = cardMap.get(rowIndex);
                if (card) {
                    cardWrapper.appendChild(card);
                }
            });
        }
        
        // Tab switching functionality
        function switchTab(tabIndex) {
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach((content, index) => {
                content.classList.remove('active');
                // Use CSS classes instead of inline styles to allow media queries to work
                if (index === tabIndex) {
                    content.classList.add('active');
                }
            });
            
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach((button, index) => {
                button.classList.remove('active');
                if (index === tabIndex) {
                    button.classList.add('active');
                }
            });
        }
        
        // Handle Enter key in filter input
        function handleFilterKeyPress(event, prefix) {
            if (event.key === 'Enter') {
                addFilter(prefix);
            }
        }
        
        // Toggle filter section and column names
        function toggleFilterSection(prefix) {
            const filterSection = document.getElementById(prefix + 'filterSection');
            const button = document.getElementById(prefix + 'toggleFilterBtn');
            const tableId = prefix + 'resultsTable';
            // Only toggle filterable display on column headers, not group headers
            const headerRow = document.querySelector('#' + tableId + ' tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : document.querySelectorAll('#' + tableId + ' th');
            
            if (!filterSection || !button) return;
            
            // Toggle filter section
            const isExpanded = filterSection.classList.contains('expanded');
            
            if (isExpanded) {
                // Collapse: hide filter section and show display names (or short/long for grouped columns)
                filterSection.classList.remove('expanded');
                headers.forEach(th => {
                    th.classList.remove('showing-filterable');
                });
                button.textContent = '🔍 Filter';
                button.title = 'Show filter options and filterable column names';
            } else {
                // Expand: show filter section and show filterable names
                filterSection.classList.add('expanded');
                headers.forEach(th => {
                    th.classList.add('showing-filterable');
                });
                button.textContent = '✖️ Hide Filter';
                button.title = 'Hide filter options and show display column names';
            }
        }

        // Toggle hidden columns visibility
        function toggleHiddenColumns(prefix) {
            const wrapperId = prefix + 'tableWrapper';
            const wrapper = document.getElementById(wrapperId);
            const btn = document.getElementById(prefix + 'toggleHiddenBtn');
            if (!wrapper || !btn) return;
            const willShow = wrapper.classList.contains('hide-hidden');
            if (willShow) {
                wrapper.classList.remove('hide-hidden');
                btn.textContent = '🙈 Hide hidden columns';
                btn.title = 'Hide the default-hidden columns';
            } else {
                wrapper.classList.add('hide-hidden');
                btn.textContent = '👁️ Show hidden columns';
                btn.title = 'Show the default-hidden columns';
            }
            // Reapply striping after column visibility changes
            applyRowStriping(prefix);
            applyColumnStriping(prefix);
        }
        
        // Update URL with current filters
        function updateURL() {
            const params = new URLSearchParams();
            
            // Add filter logic for calls (only if not default AND)
            if (filterLogic['calls'] && filterLogic['calls'] !== 'AND') {
                params.set('calls_filterLogic', filterLogic['calls']);
            }
            
            // Add filters for calls
            if (activeFilters['calls'] && activeFilters['calls'].length > 0) {
                const filterStrings = activeFilters['calls'].map(f => {
                    let filterStr = f.field;
                    if (f.operator) {
                        filterStr += ' ' + f.operator;
                        if (f.value !== null) {
                            filterStr += ' ' + f.value;
                        }
                    }
                    return filterStr;
                });
                params.set('calls_filters', filterStrings.join('|'));
            }
            
            // Add filter logic for puts (only if not default AND)
            if (filterLogic['puts'] && filterLogic['puts'] !== 'AND') {
                params.set('puts_filterLogic', filterLogic['puts']);
            }
            
            // Add filters for puts
            if (activeFilters['puts'] && activeFilters['puts'].length > 0) {
                const filterStrings = activeFilters['puts'].map(f => {
                    let filterStr = f.field;
                    if (f.operator) {
                        filterStr += ' ' + f.operator;
                        if (f.value !== null) {
                            filterStr += ' ' + f.value;
                        }
                    }
                    return filterStr;
                });
                params.set('puts_filters', filterStrings.join('|'));
            }
            
            // Update URL without reloading page
            const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
            window.history.replaceState({}, '', newURL);
        }
        
        // Load filters from URL
        function loadFiltersFromURL() {
            const params = new URLSearchParams(window.location.search);
            
            // Load filters for calls
            loadFiltersForPrefix('calls', params);
            
            // Load filters for puts
            loadFiltersForPrefix('puts', params);
        }
        
        // Helper function to load filters for a specific prefix
        function loadFiltersForPrefix(prefix, params) {
            const errorDiv = document.getElementById(prefix + 'filterError');
            if (errorDiv) {
                errorDiv.textContent = '';
                errorDiv.style.display = 'none';
            }
            
            initStateForPrefix(prefix);
            const tableId = prefix + 'resultsTable';
            
            // Load filter logic
            const urlFilterLogic = params.get(prefix + '_filterLogic');
            if (urlFilterLogic && (urlFilterLogic === 'AND' || urlFilterLogic === 'OR')) {
                filterLogic[prefix] = urlFilterLogic;
                // Update radio button
                const radio = document.querySelector(`input[name="${prefix}filterLogic"][value="${filterLogic[prefix]}"]`);
                if (radio) {
                    radio.checked = true;
                }
            }
            
            // Load filters
            const filtersParam = params.get(prefix + '_filters');
            if (filtersParam) {
                const filterStrings = filtersParam.split('|');
                activeFilters[prefix] = [];
                const validationErrors = [];
                
                for (const filterStr of filterStrings) {
                    if (filterStr.trim()) {
                        const filter = parseFilterExpression(filterStr.trim(), tableId);
                        if (filter) {
                            // Validate filter before adding
                            const errors = validateFilter(filter, tableId);
                            if (errors.length > 0) {
                                validationErrors.push(...errors);
                            } else {
                                activeFilters[prefix].push(filter);
                            }
                        }
                    }
                }
                
                // Show validation errors if any
                if (validationErrors.length > 0 && errorDiv) {
                    errorDiv.textContent = 'Warning: Some filters from URL could not be loaded: ' + validationErrors.join(' ');
                    errorDiv.style.display = 'block';
                }
                
                // If filters were loaded from URL, expand the filter section
                if (activeFilters[prefix].length > 0) {
                    const filterSection = document.getElementById(prefix + 'filterSection');
                    const button = document.getElementById(prefix + 'toggleFilterBtn');
                    // Only show filterable names on column headers, not group headers
                    const headerRow = document.querySelector('#' + tableId + ' tr.column-header-row');
                    const headers = headerRow ? headerRow.querySelectorAll('th') : document.querySelectorAll('#' + tableId + ' th');
                    
                    if (filterSection && button) {
                        filterSection.classList.add('expanded');
                        headers.forEach(th => {
                            th.classList.add('showing-filterable');
                        });
                        button.textContent = '✖️ Hide Filter';
                        button.title = 'Hide filter options and show display column names';
                    }
                }
                
                updateFilterDisplay(prefix);
                applyFilters(prefix);
            }
        }
        
        // Calculate and display time ago
        function updateTimeAgo() {
            const timeAgoEl = document.getElementById('timeAgo');
            const generatedTimeEl = document.getElementById('generatedTime');
            if (!timeAgoEl || !generatedTimeEl) return;
            
            // Get the ISO timestamp from the data-generated attribute on the paragraph
            const generatedTimeStr = generatedTimeEl.getAttribute('data-generated');
            if (!generatedTimeStr || generatedTimeStr.trim() === '' || generatedTimeStr === '+') {
                // Silently return if no valid timestamp
                return;
            }
            
            try {
                // Clean up the timestamp string - handle timezone offsets properly
                let cleanTimestamp = generatedTimeStr.trim();
                // If it's just a "+" or malformed, skip
                if (cleanTimestamp === '+' || cleanTimestamp.length < 10) {
                    return;
                }
                
                const generatedTime = new Date(cleanTimestamp);
                // Check if date is valid
                if (isNaN(generatedTime.getTime())) {
                    // Silently fail instead of logging error for malformed dates
                    timeAgoEl.textContent = '';
                    return;
                }
                
                const now = new Date();
                const diffMs = now - generatedTime;
                
                // Handle negative differences (future dates) or invalid calculations
                if (isNaN(diffMs) || diffMs < 0) {
                    timeAgoEl.textContent = '';
                    return;
                }
                
                const diffSeconds = Math.floor(diffMs / 1000);
                const diffMinutes = Math.floor(diffSeconds / 60);
                const diffHours = Math.floor(diffMinutes / 60);
                const diffDays = Math.floor(diffHours / 24);
                
                let timeAgoText = '';
                if (diffSeconds < 60) {
                    timeAgoText = `(${diffSeconds} second${diffSeconds !== 1 ? 's' : ''} ago)`;
                } else if (diffMinutes < 60) {
                    timeAgoText = `(${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago)`;
                } else if (diffHours < 24) {
                    timeAgoText = `(${diffHours} hour${diffHours !== 1 ? 's' : ''} ago)`;
                } else if (diffDays < 7) {
                    timeAgoText = `(${diffDays} day${diffDays !== 1 ? 's' : ''} ago)`;
                } else {
                    const diffWeeks = Math.floor(diffDays / 7);
                    // Validate diffWeeks is a valid number
                    if (isNaN(diffWeeks) || diffWeeks < 0) {
                        timeAgoEl.textContent = '';
                        return;
                    }
                    timeAgoText = `(${diffWeeks} week${diffWeeks !== 1 ? 's' : ''} ago)`;
                }
                
                timeAgoEl.textContent = timeAgoText;
            } catch (e) {
                console.error('Error calculating time ago:', e);
                timeAgoEl.textContent = '';
            }
        }
        
        // Toggle card details (expand/collapse)
        function toggleCardDetails(prefix, rowIndex) {
            const detailsEl = document.getElementById(prefix + 'cardDetails_' + rowIndex);
            const toggleBtn = document.getElementById(prefix + 'cardToggle_' + rowIndex);
            if (!detailsEl || !toggleBtn) return;
            
            const isExpanded = detailsEl.classList.contains('expanded');
            if (isExpanded) {
                detailsEl.classList.remove('expanded');
                toggleBtn.classList.remove('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Show More Details';
            } else {
                detailsEl.classList.add('expanded');
                toggleBtn.classList.add('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Hide Details';
            }
        }
        
        // Sync card visibility with table row visibility (for filtering)
        function syncCardVisibility(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            const tbody = table ? table.querySelector('tbody') : null;
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const cardWrapperId = prefix + 'cardWrapper';
            const cardWrapper = document.getElementById(cardWrapperId);
            const cards = cardWrapper ? Array.from(cardWrapper.querySelectorAll('.data-card')) : [];
            
            // Create a map of row index to card
            const cardMap = new Map();
            cards.forEach(card => {
                const rowIndex = card.getAttribute('data-row-index');
                if (rowIndex !== null) {
                    cardMap.set(rowIndex, card);
                }
            });
            
            // Sync visibility based on row index
            rows.forEach((row) => {
                const rowIndex = row.getAttribute('data-row-index');
                if (rowIndex !== null) {
                    const card = cardMap.get(rowIndex);
                    if (card) {
                        const isVisible = row.style.display !== 'none';
                        if (isVisible) {
                            card.classList.remove('hidden');
                        } else {
                            card.classList.add('hidden');
                        }
                    }
                }
            });
            
            // Update visible count for cards
            const visibleCards = cards.filter(card => !card.classList.contains('hidden'));
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleCards.length;
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize state for both prefixes
            initStateForPrefix('calls');
            initStateForPrefix('puts');
            
            switchTab(0);
            
            // Initialize column maps for both tables
            initColumnMap('calls');
            initColumnMap('puts');
            
            // Update filter displays
            updateFilterDisplay('calls');
            updateFilterDisplay('puts');
            
            // Load filters from URL
            loadFiltersFromURL();
            
            updateTimeAgo(); // Update time ago on page load
            // Update time ago every minute
            setInterval(updateTimeAgo, 60000);
            
            // Ensure hidden columns are hidden on load for both tables
            ['calls', 'puts'].forEach(prefix => {
                const wrapper = document.getElementById(prefix + 'tableWrapper');
                const toggleBtn = document.getElementById(prefix + 'toggleHiddenBtn');
                const ensureHidden = wrapper && !wrapper.classList.contains('hide-hidden');
                if (ensureHidden) {
                    wrapper.classList.add('hide-hidden');
                }
                if (toggleBtn) {
                    if (wrapper && wrapper.classList.contains('hide-hidden')) {
                        toggleBtn.textContent = '👁️ Show hidden columns';
                        toggleBtn.title = 'Show the default-hidden columns';
                    } else {
                        toggleBtn.textContent = '🙈 Hide hidden columns';
                        toggleBtn.title = 'Hide the default-hidden columns';
                    }
                }
                
                // Apply initial row and column striping
                applyRowStriping(prefix);
                applyColumnStriping(prefix);
                
                // Sync card visibility on load
                syncCardVisibility(prefix);
                
                // Default sort by net_premium descending
                const netPremiumColIndex = findColumnIndex('net_premium', prefix + 'resultsTable');
                if (netPremiumColIndex >= 0) {
                    // Call sortTable twice: first sets to asc, second toggles to desc
                    sortTable(prefix, netPremiumColIndex);
                    sortTable(prefix, netPremiumColIndex);
                }
            });
        });
"""
