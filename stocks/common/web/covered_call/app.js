        // API Configuration (injected by HTML)
        const API_CONFIG = window.API_CONFIG || {};
        
        // API state
        let apiDataCache = {};
        let apiLoading = {};
        
        // Fetch data from API
        async function fetchDataFromAPI(prefix, optionType = null) {
            const cacheKey = prefix + (optionType || 'all');
            if (apiLoading[cacheKey]) {
                return; // Already loading
            }
            
            // Determine option type
            if (!optionType) {
                optionType = prefix === 'calls' ? 'call' : (prefix === 'puts' ? 'put' : 'all');
            }
            
            apiLoading[cacheKey] = true;
            
            // Show loading indicator
            showLoadingIndicator(prefix);
            
            try {
                // Build query parameters
                const params = new URLSearchParams({
                    option_type: optionType
                });
                
                // Add filters if any - use pipe-separated format
                if (activeFilters[prefix] && activeFilters[prefix].length > 0) {
                    const filterStrings = activeFilters[prefix].map(f => {
                        let filterStr = f.field;
                        if (f.operator) {
                            filterStr += ' ' + f.operator;
                            if (f.value !== null) {
                                // Use valueStr if available (preserves % sign), otherwise use value
                                const valueToUse = (f.valueStr !== undefined) ? f.valueStr : String(f.value);
                                filterStr += ' ' + valueToUse;
                            }
                        }
                        return filterStr;
                    });
                    // Use prefix-specific parameter name (calls_filters or puts_filters)
                    params.set(prefix + '_filters', filterStrings.join('|'));
                    // Add filter logic with prefix-specific parameter name
                    const logic = filterLogic[prefix] || 'AND';
                    if (logic !== 'AND') {
                        params.set(prefix + '_filterLogic', logic);
                    }
                }
                
                // Add sorting
                if (currentSortColumn[prefix] !== undefined && currentSortColumn[prefix] >= 0) {
                    const tableId = prefix + 'resultsTable';
                    const table = document.getElementById(tableId);
                    if (table) {
                        const headerRow = table.querySelector('tr.column-header-row');
                        const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
                        if (headers[currentSortColumn[prefix]]) {
                            let colName = headers[currentSortColumn[prefix]].getAttribute('data-filterable-name') || 
                                          headers[currentSortColumn[prefix]].textContent.trim();
                            // When sorting on tkr_info, sort by risk_score instead
                            if (colName === 'tkr_info') {
                                colName = 'risk_score';
                            }
                            params.set('sort', colName);
                            params.set('sort_direction', sortDirection[prefix][currentSortColumn[prefix]] || 'desc');
                        }
                    }
                }
                
                // Use relative URL - same host as the HTML page
                const url = `/stock_info/api/covered_calls/data?${params.toString()}`;
                const response = await fetch(url);
                
                if (!response.ok) {
                    throw new Error(`API error: ${response.status} ${response.statusText}`);
                }
                
                const data = await response.json();
                apiDataCache[cacheKey] = data;
                
                // Render table with API data
                renderTableFromAPI(prefix, data);
                
            } catch (error) {
                console.error('Error fetching data from API:', error);
                const tableId = prefix + 'resultsTable';
                const tbody = document.querySelector(`#${tableId} tbody`);
                if (tbody) {
                    tbody.innerHTML = `<tr><td colspan="100%" style="text-align: center; padding: 20px; color: red;">Error loading data: ${error.message}</td></tr>`;
                }
            } finally {
                apiLoading[cacheKey] = false;
                // Hide loading indicator
                hideLoadingIndicator(prefix);
            }
        }
        
        // Fetch comprehensive analysis from API
        async function fetchAnalysisFromAPI() {
            const cacheKey = 'analysis';
            if (apiLoading[cacheKey]) {
                return; // Already loading
            }
            
            apiLoading[cacheKey] = true;
            
            // Get analysis type preference from checkbox
            const useGeminiCheckbox = document.getElementById('useGeminiAnalysis');
            const useGemini = useGeminiCheckbox ? useGeminiCheckbox.checked : false;
            
            // Update status
            const statusEl = document.getElementById('analysisStatus');
            if (statusEl) {
                if (useGemini) {
                    statusEl.textContent = '⏳ Generating Gemini AI analysis (this may take 1-2 minutes)...';
                    statusEl.style.color = '#ff9800';
                } else {
                    statusEl.textContent = 'Loading rule-based analysis...';
                    statusEl.style.color = '#666';
                }
            }
            
            // Show loading indicator in content area with different messages for Gemini
            const analysisContent = document.getElementById('analysisContent');
            if (analysisContent) {
                if (useGemini) {
                    analysisContent.innerHTML = `
                        <div class="loading-indicator" style="text-align: center; padding: 40px;">
                            <div class="spinner" style="margin: 0 auto 20px;"></div>
                            <div class="loading-text" style="font-size: 16px; font-weight: 500; color: #667eea; margin-bottom: 10px;">
                                🤖 Generating Gemini AI Analysis
                            </div>
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                                This may take 1-2 minutes...
                            </div>
                            <div style="font-size: 12px; color: #999;">
                                Analyzing ${useGemini ? 'filtered' : 'all'} data with AI
                            </div>
                            <div style="margin-top: 20px; font-size: 11px; color: #aaa;">
                                <div class="pulse-dots">
                                    <span>.</span><span>.</span><span>.</span>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    analysisContent.innerHTML = '<div class="loading-indicator"><div class="spinner"></div><div class="loading-text">Loading rule-based analysis...</div></div>';
                }
            }
            
            try {
                // Build query parameters - use filters from both calls and puts
                // Note: source parameter is required for analysis endpoint
                const params = new URLSearchParams({
                    option_type: 'all',  // Get all data for comprehensive analysis
                    use_gemini: useGemini ? 'true' : 'false'  // Use checkbox value
                });
                
                // Add source parameter - use configured source or default
                const source = API_CONFIG.csv_source || '~/Downloads/results.csv';
                params.set('source', source);
                
                // Combine filters from both calls and puts
                const allFilters = [];
                if (activeFilters['calls'] && activeFilters['calls'].length > 0) {
                    const callsFilterStrings = activeFilters['calls'].map(f => {
                        let filterStr = f.field;
                        if (f.operator) {
                            filterStr += ' ' + f.operator;
                            if (f.value !== null) {
                                const valueToUse = (f.valueStr !== undefined) ? f.valueStr : String(f.value);
                                filterStr += ' ' + valueToUse;
                            }
                        }
                        return filterStr;
                    });
                    if (callsFilterStrings.length > 0) {
                        params.set('calls_filters', callsFilterStrings.join('|'));
                        const logic = filterLogic['calls'] || 'AND';
                        if (logic !== 'AND') {
                            params.set('calls_filterLogic', logic);
                        }
                    }
                }
                
                if (activeFilters['puts'] && activeFilters['puts'].length > 0) {
                    const putsFilterStrings = activeFilters['puts'].map(f => {
                        let filterStr = f.field;
                        if (f.operator) {
                            filterStr += ' ' + f.operator;
                            if (f.value !== null) {
                                const valueToUse = (f.valueStr !== undefined) ? f.valueStr : String(f.value);
                                filterStr += ' ' + valueToUse;
                            }
                        }
                        return filterStr;
                    });
                    if (putsFilterStrings.length > 0) {
                        params.set('puts_filters', putsFilterStrings.join('|'));
                        const logic = filterLogic['puts'] || 'AND';
                        if (logic !== 'AND') {
                            params.set('puts_filterLogic', logic);
                        }
                    }
                }
                
                // Use relative URL - same host as the HTML page
                const url = `/stock_info/api/covered_calls/analysis?${params.toString()}`;
                
                // For Gemini analysis, use a much longer timeout (6 minutes = 360 seconds)
                // since Gemini can take up to 5 minutes per option type
                const timeoutMs = useGemini ? 360000 : 60000; // 6 minutes for Gemini, 1 minute for rule-based
                
                // Create an AbortController for timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
                
                try {
                    const response = await fetch(url, {
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                    
                    if (!response.ok) {
                        throw new Error(`API error: ${response.status} ${response.statusText}`);
                    }
                
                    const htmlContent = await response.text();
                    
                    // Update analysis content area (not the whole tab, preserve controls)
                    if (analysisContent) {
                        analysisContent.innerHTML = htmlContent;
                    }
                    
                    // Update status
                    if (statusEl) {
                        statusEl.textContent = useGemini ? '✓ Gemini AI analysis loaded' : '✓ Rule-based analysis loaded';
                        statusEl.style.color = '#28a745';
                    }
                } catch (fetchError) {
                    clearTimeout(timeoutId);
                    if (fetchError.name === 'AbortError') {
                        throw new Error('Request timeout: Analysis took too long. Please try again or use rule-based analysis.');
                    }
                    throw fetchError;
                }
                
            } catch (error) {
                console.error('Error fetching analysis from API:', error);
                if (analysisContent) {
                    let errorMsg = error.message;
                    if (errorMsg.includes('timeout') || errorMsg.includes('504')) {
                        errorMsg = 'Request timed out. Gemini analysis can take up to 5 minutes. Please try again or check server logs.';
                    }
                    analysisContent.innerHTML = `<div class="error" style="padding: 20px; color: red;">Error loading comprehensive analysis: ${errorMsg}</div>`;
                }
                if (statusEl) {
                    statusEl.textContent = '✗ Error loading analysis';
                    statusEl.style.color = '#dc3545';
                }
            } finally {
                apiLoading[cacheKey] = false;
            }
        }
        
        // Global function to load analysis (called by button)
        function loadComprehensiveAnalysis() {
            fetchAnalysisFromAPI();
        }
        
        // Render table from API data
        function renderTableFromAPI(prefix, apiData) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            if (!tbody) return;
            
            const data = apiData.data || [];
            const metadata = apiData.metadata || {};
            
            // Update data source timestamp if available
            if (metadata.data_source_timestamp) {
                updateDataSourceTimestamp(metadata.data_source_timestamp);
            }
            
            // Update stats
            const totalCountEl = document.getElementById(prefix + 'totalCount');
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (totalCountEl) totalCountEl.textContent = metadata.total_count || 0;
            if (visibleCountEl) visibleCountEl.textContent = metadata.filtered_count || data.length;
            
            // Clear existing rows
            tbody.innerHTML = '';
            
            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="100%" style="text-align: center; padding: 20px;">No data available.</td></tr>';
                return;
            }
            
            // Get column order from table headers
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            const columnOrder = [];
            const columnMetadata = []; // Store metadata for each column
            
            headers.forEach(th => {
                const colName = th.getAttribute('data-filterable-name') || th.textContent.trim();
                columnOrder.push(colName);
                
                // Store metadata about hidden columns
                const isHidden = th.classList.contains('is-hidden-col');
                const isAlwaysHidden = th.classList.contains('always-hidden');
                columnMetadata.push({
                    name: colName,
                    isHidden: isHidden,
                    isAlwaysHidden: isAlwaysHidden
                });
            });
            
            // Build a map of normalized column names from API data for better matching
            const apiColumnMap = {};
            if (data.length > 0) {
                Object.keys(data[0]).forEach(key => {
                    const normalized = key.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_');
                    if (!apiColumnMap[normalized]) {
                        apiColumnMap[normalized] = [];
                    }
                    apiColumnMap[normalized].push(key);
                });
                // Debug: log available columns from API
                console.log('API columns:', Object.keys(data[0]));
                console.log('Table columns:', columnOrder);
            }
            
            // Helper to find matching column name in API data
            function findApiColumnName(headerColName) {
                // Try exact match first
                if (data.length > 0 && data[0].hasOwnProperty(headerColName)) {
                    return headerColName;
                }
                
                // Explicit mappings for known column name variations
                // Maps HTML table column names to API column names
                const columnNameMappings = {
                    'opt_prem.': 'option_premium',
                    'opt_prem': 'option_premium',
                    'l_prem': 'long_option_premium',
                    'l_opt_prem': 'long_option_premium',
                    'premium_ratio_pct': 'premium_ratio_pct',  // Ensure this maps correctly
                    'premium_ratio_p': 'premium_ratio_pct',    // Handle truncation
                };
                
                // Check explicit mappings first
                if (columnNameMappings[headerColName] && data.length > 0 && data[0].hasOwnProperty(columnNameMappings[headerColName])) {
                    return columnNameMappings[headerColName];
                }
                
                // Special handling for latest_option_writets: API may return latest_opt_ts
                // (prepare_dataframe_for_display should rename it, but if it doesn't, try both)
                if (headerColName === 'latest_option_writets') {
                    if (data.length > 0 && data[0].hasOwnProperty('latest_opt_ts')) {
                        return 'latest_opt_ts';
                    }
                    if (data.length > 0 && data[0].hasOwnProperty('latest_option_writets')) {
                        return 'latest_option_writets';
                    }
                }
                
                // Try case-insensitive match (normalize by removing dots and spaces)
                const normalized = headerColName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_').replace(/\./g, '');
                if (apiColumnMap[normalized] && apiColumnMap[normalized].length > 0) {
                    return apiColumnMap[normalized][0]; // Use first match
                }
                
                // Try partial match with normalized names
                for (const apiKey in data[0] || {}) {
                    const normalizedApiKey = apiKey.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_').replace(/\./g, '');
                    if (normalizedApiKey === normalized) {
                        return apiKey;
                    }
                }
                
                return null;
            }
            
            // Helper function to format age in seconds to human-readable format
            function formatAgeSeconds(ageSeconds) {
                if (ageSeconds === null || ageSeconds === undefined || ageSeconds === '') {
                    return '';
                }
                const age = parseFloat(ageSeconds);
                if (isNaN(age) || age < 0) return '';
                if (age === 0) return '0 secs';
                if (age < 60) return Math.floor(age) + ' secs';
                if (age < 3600) return (age / 60).toFixed(1) + ' mins';
                if (age < 86400) return (age / 3600).toFixed(1) + ' hrs';
                return (age / 86400).toFixed(1) + ' days';
            }
            
            // Helper function to format numeric value based on column type
            function formatCellValue(value, colName) {
                if (value === null || value === undefined || value === '') {
                    return { display: '', raw: null };
                }
                
                const normalizedCol = colName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_');
                const numValue = typeof value === 'number' ? value : parseFloat(value);
                const isNumeric = !isNaN(numValue) && value !== '';
                
                // Age columns (latest_option_writets, latest_opt_ts) - format as human-readable
                if ((normalizedCol.includes('writets') || normalizedCol.includes('latest_opt') || 
                     normalizedCol.includes('age')) && isNumeric) {
                    const ageDisplay = formatAgeSeconds(numValue);
                    return {
                        display: ageDisplay,
                        raw: String(numValue)
                    };
                }
                
                // Percentage columns
                // NOTE: Handle percentage BEFORE currency so columns like 'premium_ratio_pct'
                //       are formatted as percentages, not dollars.
                // Check for percentage indicators (pct, percent, or ends with %)
                // Also explicitly check for premium_ratio_pct to ensure it's always treated as percentage
                if (isNumeric && (normalizedCol === 'premium_ratio_pct' || 
                    normalizedCol.includes('_pct') || normalizedCol.includes('_percent') || 
                    normalizedCol.endsWith('_pct') || normalizedCol.endsWith('_percent') ||
                    normalizedCol.includes('pct') || normalizedCol.includes('percent') || 
                    normalizedCol.endsWith('%'))) {
                    return {
                        display: numValue.toFixed(2) + '%',
                        raw: String(numValue)
                    };
                }
                
                // Currency columns (price, premium, strike, cost, total)
                // BUT exclude premium_ratio_pct which should already be handled above
                // Also check if current_price contains price_with_change format (has +$ or -$)
                if (normalizedCol === 'current_price' && typeof value === 'string' && 
                    (value.includes('+$') || value.includes('-$') || value.includes('(+') || value.includes('(-'))) {
                    // This is already formatted as price_with_change, return as-is
                    return {
                        display: value,
                        raw: value
                    };
                }
                
                if (isNumeric && normalizedCol !== 'premium_ratio_pct' &&
                    (normalizedCol.includes('price') || normalizedCol.includes('premium') || 
                    normalizedCol.includes('prem') || normalizedCol.includes('strike') || 
                    normalizedCol.includes('cost') || normalizedCol.includes('total'))) {
                    return {
                        display: '$' + numValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
                        raw: String(numValue)
                    };
                }
                
                // Integer columns (volume, contracts, days, count)
                if (isNumeric && (normalizedCol.includes('volume') || normalizedCol.includes('contracts') || 
                    normalizedCol.includes('cnt') || (normalizedCol.includes('days') && !normalizedCol.includes('writets')) || 
                    normalizedCol.includes('count'))) {
                    return {
                        display: Math.floor(numValue).toLocaleString('en-US'),
                        raw: String(Math.floor(numValue))
                    };
                }
                
                // Trade quality - 2 decimal places
                if (isNumeric && normalizedCol === 'trade_quality') {
                    return {
                        display: numValue.toFixed(2),
                        raw: String(numValue)
                    };
                }
                
                // Date columns - check if it's a date string (YYYY-MM-DD or ISO format)
                if (typeof value === 'string' && (normalizedCol.includes('expiration') || normalizedCol.includes('exp_date') || normalizedCol.includes('date'))) {
                    // Extract just the date portion (YYYY-MM-DD) from timestamp strings
                    const dateMatch = value.match(/^(\d{4}-\d{2}-\d{2})/);
                    if (dateMatch) {
                        return {
                            display: dateMatch[1], // Just the date part (YYYY-MM-DD)
                            raw: value
                        };
                    }
                    // If it's already a formatted date string, keep it
                    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
                        return {
                            display: value,
                            raw: value
                        };
                    }
                }
                
                // Default: return as string
                return {
                    display: String(value),
                    raw: isNumeric ? String(numValue) : String(value)
                };
            }
            
            // Render rows
            data.forEach((row, rowIdx) => {
                const tr = document.createElement('tr');
                tr.setAttribute('data-row-index', rowIdx);
                
                columnOrder.forEach((colName, colIdx) => {
                    const td = document.createElement('td');
                    const colMeta = columnMetadata[colIdx] || {};
                    
                    // Normalize column name for comparison
                    const normalizedCol = colName.toLowerCase().replace(/\s+/g, '_');
                    
                    // Find matching column name in API data
                    const apiColName = findApiColumnName(colName);
                    const value = apiColName ? row[apiColName] : null;
                    
                    // Debug missing columns (only log once per column)
                    if (!apiColName && rowIdx === 0) {
                        console.warn(`Column "${colName}" not found in API data. Available:`, Object.keys(row));
                    }
                    
                    // Special handling for tkr_info column - display as table with 4 rows (check before formatting)
                    // Check both normalized name and original column name
                    const isTkrInfoCol = normalizedCol === 'tkr_info' || 
                                         colName.toLowerCase().includes('tkr_info') ||
                                         colName.toLowerCase().replace(/\s+/g, '_') === 'tkr_info';
                    
                    if (isTkrInfoCol) {
                        // Helper function to safely parse numeric values, handling empty strings and NaN
                        function safeParseFloat(value) {
                            if (value === undefined || value === null || value === '' || value === 'NaN' || String(value).trim() === '') {
                                return null;
                            }
                            const parsed = parseFloat(value);
                            return (isNaN(parsed) || !isFinite(parsed)) ? null : parsed;
                        }
                        
                        // Get IV metrics from row data directly (not from column value)
                        // These columns should be in the API response
                        // Handle both full and truncated column names (e.g., iv_recommendation vs iv_recommendati)
                        const riskScore = safeParseFloat(row['risk_score']);
                        const ivRank30 = safeParseFloat(row['iv_rank_30']);
                        const ivRank90 = safeParseFloat(row['iv_rank_90']);
                        // Try multiple possible column names (full, truncated, compact)
                        const ivRecommendationRaw = (row['iv_recommendation'] || row['iv_recommendati'] || row['iv_rec'] || null);
                        const ivRecommendation = ivRecommendationRaw ? String(ivRecommendationRaw).trim() : null;
                        const rollYield = safeParseFloat(row['roll_yield']);
                        
                        // Debug logging (only for first row to avoid spam)
                        if (rowIdx === 0 && colIdx === 1) {
                            console.log('=== tkr_info column detected ===');
                            console.log('Column name:', colName);
                            console.log('Normalized col:', normalizedCol);
                            console.log('Column index:', colIdx);
                            console.log('Row index:', rowIdx);
                            console.log('All row keys:', Object.keys(row));
                            console.log('IV-related keys:', Object.keys(row).filter(k => k.includes('risk') || k.includes('iv') || k.includes('roll') || k.includes('recommend')));
                            console.log('Raw values:', {
                                risk_score: row['risk_score'],
                                iv_rank_30: row['iv_rank_30'],
                                iv_rank_90: row['iv_rank_90'],
                                iv_recommendation: row['iv_recommendation'],
                                iv_recommendati: row['iv_recommendati'],
                                iv_rec: row['iv_rec'],
                                roll_yield: row['roll_yield']
                            });
                            console.log('Parsed values:', {
                                riskScore: riskScore,
                                ivRank30: ivRank30,
                                ivRank90: ivRank90,
                                ivRecommendation: ivRecommendation,
                                rollYield: rollYield
                            });
                            console.log('===============================');
                        }
                        
                        // Create compact nested table
                        const table = document.createElement('table');
                        table.style.cssText = 'width:100%; border-collapse:collapse; margin:0; padding:0; line-height:1.2; font-size:0.85em;';
                        
                        // Risk Score row
                        const riskRow = table.insertRow();
                        const riskLabelCell = riskRow.insertCell();
                        riskLabelCell.textContent = 'Risk:';
                        riskLabelCell.style.cssText = 'padding:2px 4px 2px 0; margin:0; font-weight:600; border:none; text-align:left; white-space:nowrap;';
                        const riskValueCell = riskRow.insertCell();
                        riskValueCell.textContent = (riskScore !== null && !isNaN(riskScore)) ? riskScore.toFixed(1) : 'N/A';
                        riskValueCell.style.cssText = 'padding:2px 0; margin:0; border:none; text-align:right;';
                        
                        // IV Rank row (30/90)
                        const rankRow = table.insertRow();
                        const rankLabelCell = rankRow.insertCell();
                        rankLabelCell.textContent = 'IV Rank:';
                        rankLabelCell.style.cssText = 'padding:2px 4px 2px 0; margin:0; font-weight:600; border:none; text-align:left; white-space:nowrap;';
                        const rankValueCell = rankRow.insertCell();
                        if (ivRank30 !== null && !isNaN(ivRank30) && ivRank90 !== null && !isNaN(ivRank90)) {
                            rankValueCell.textContent = `${ivRank30.toFixed(1)}/${ivRank90.toFixed(1)}`;
                        } else if (ivRank30 !== null && !isNaN(ivRank30)) {
                            rankValueCell.textContent = `${ivRank30.toFixed(1)}/-`;
                        } else if (ivRank90 !== null && !isNaN(ivRank90)) {
                            rankValueCell.textContent = `-/${ivRank90.toFixed(1)}`;
                        } else {
                            rankValueCell.textContent = 'N/A';
                        }
                        rankValueCell.style.cssText = 'padding:2px 0; margin:0; border:none; text-align:right;';
                        
                        // IV Recommendation row
                        const recRow = table.insertRow();
                        const recLabelCell = recRow.insertCell();
                        recLabelCell.textContent = 'IV Rec:';
                        recLabelCell.style.cssText = 'padding:2px 4px 2px 0; margin:0; font-weight:600; border:none; text-align:left; white-space:nowrap;';
                        const recValueCell = recRow.insertCell();
                        recValueCell.textContent = ivRecommendation || 'N/A';
                        recValueCell.style.cssText = 'padding:2px 0; margin:0; border:none; text-align:right; font-size:0.9em;';
                        
                        // Roll Yield row
                        const rollRow = table.insertRow();
                        const rollLabelCell = rollRow.insertCell();
                        rollLabelCell.textContent = 'Roll Yield:';
                        rollLabelCell.style.cssText = 'padding:2px 4px 2px 0; margin:0; font-weight:600; border:none; text-align:left; white-space:nowrap;';
                        const rollValueCell = rollRow.insertCell();
                        if (rollYield !== null && !isNaN(rollYield)) {
                            // Parse roll yield - it might be a string like "2.5%" or a number
                            let rollYieldValue = rollYield;
                            if (typeof rollYield === 'string' && rollYield.endsWith('%')) {
                                rollYieldValue = parseFloat(rollYield.replace('%', ''));
                            }
                            if (!isNaN(rollYieldValue) && isFinite(rollYieldValue)) {
                                rollValueCell.textContent = rollYieldValue.toFixed(2) + '%';
                            } else {
                                rollValueCell.textContent = 'N/A';
                            }
                        } else {
                            rollValueCell.textContent = 'N/A';
                        }
                        rollValueCell.style.cssText = 'padding:2px 0; margin:0; border:none; text-align:right;';
                        
                        td.innerHTML = '';
                        td.appendChild(table);
                        td.style.cssText = 'padding:4px 6px; vertical-align:top;';
                        
                        // Debug: verify table was created
                        if (rowIdx === 0 && colIdx === 1) {
                            console.log('Table created, rows:', table.rows.length);
                            console.log('Table HTML:', td.innerHTML.substring(0, 200));
                        }
                        
                        // Set raw value for filtering/sorting (for tkr_info, use risk_score as raw value)
                        td.setAttribute('data-raw', riskScore !== null ? riskScore : '');
                        
                        // Add CSS classes for hidden columns
                        if (colMeta.isHidden) {
                            td.classList.add('is-hidden-col');
                        }
                        if (colMeta.isAlwaysHidden) {
                            td.classList.add('always-hidden');
                        }
                        
                        tr.appendChild(td);
                        return; // Skip the rest of the column processing for tkr_info
                    }
                    
                    // Format the value (handle null/undefined gracefully)
                    const formatted = formatCellValue(value !== undefined ? value : null, colName);
                    
                    // Set display value
                    if (formatted.display) {
                            // Special handling for current_price with change data - display as compact table
                            if (normalizedCol === 'current_price' && typeof formatted.display === 'string' && 
                            (formatted.display.includes('+$') || formatted.display.includes('-$'))) {
                            // Parse format like "$73.56 -$4.87 (-6.21%)" or "$124.59 +$0.50 (+0.40%)"
                            const priceMatch = formatted.display.match(/^\$?([\d,]+\.\d+)\s+([\+\-]\$[\d,]+\.\d+)\s+\(([\+\-][\d\.]+%)\)/);
                            if (priceMatch) {
                                const [_, price, delta, deltaPct] = priceMatch;
                                const isPositive = delta.startsWith('+');
                                const colorClass = isPositive ? 'price-positive' : 'price-negative';
                                
                                // Create compact nested table
                                const table = document.createElement('table');
                                table.style.cssText = 'width:100%; border-collapse:collapse; margin:0; padding:0; line-height:1.1; font-size:inherit;';
                                
                                // Price row
                                const priceRow = table.insertRow();
                                const priceCell = priceRow.insertCell();
                                priceCell.textContent = '$' + price;
                                priceCell.style.cssText = 'padding:1px 0; margin:0; font-weight:500; border:none;';
                                
                                // Delta row
                                const deltaRow = table.insertRow();
                                const deltaCell = deltaRow.insertCell();
                                deltaCell.textContent = delta;
                                deltaCell.style.cssText = 'padding:1px 0; margin:0; font-size:0.85em; border:none;';
                                deltaCell.className = colorClass;
                                
                                // Delta % row
                                const deltaPctRow = table.insertRow();
                                const deltaPctCell = deltaPctRow.insertCell();
                                deltaPctCell.textContent = '(' + deltaPct + ')';
                                deltaPctCell.style.cssText = 'padding:1px 0; margin:0; font-size:0.85em; border:none;';
                                deltaPctCell.className = colorClass;
                                
                                td.innerHTML = '';
                                td.appendChild(table);
                                td.style.cssText = 'padding:2px 4px; vertical-align:top;';
                            } else {
                                td.textContent = formatted.display;
                            }
                        } else {
                            td.textContent = formatted.display;
                        }
                    }
                    
                    // Set raw value for filtering/sorting
                    if (formatted.raw !== null) {
                        td.setAttribute('data-raw', formatted.raw);
                    }
                    
                    // Add CSS classes for hidden columns
                    if (colMeta.isHidden) {
                        td.classList.add('is-hidden-col');
                    }
                    if (colMeta.isAlwaysHidden) {
                        td.classList.add('always-hidden');
                    }
                    
                    // Add color classes for price change columns
                    // Skip if current_price already has nested table (colors applied to inner cells)
                    const hasNestedTable = td.querySelector('table') !== null;
                    // Use already-defined normalizedCol from above
                    if (!hasNestedTable && (normalizedCol === 'current_price' || normalizedCol === 'price_with_change' || normalizedCol === 'change_pct') && formatted.display) {
                        const displayStr = String(formatted.display);
                        // Check for positive change (+$ or (+ or positive percentage)
                        if (displayStr.includes('+$') || displayStr.includes('(+') || (displayStr.startsWith('+') && !displayStr.startsWith('+-'))) {
                            td.classList.add('price-positive');
                        }
                        // Check for negative change (-$ or (- or negative percentage)
                        else if (displayStr.includes('-$') || displayStr.includes('(-') || (displayStr.startsWith('-') && !displayStr.startsWith('--'))) {
                            td.classList.add('price-negative');
                        }
                    }
                    
                    // Make ticker column a link with tooltip showing IV metrics
                    if (normalizedCol === 'ticker' && formatted.display) {
                        const tickerLink = document.createElement('a');
                        tickerLink.href = `/stock_info/${encodeURIComponent(formatted.display.trim())}`;
                        tickerLink.target = '_blank';
                        tickerLink.style.cssText = 'color: #667eea; text-decoration: none; font-weight: 500; cursor: help;';
                        tickerLink.textContent = formatted.display;
                        
                        // Get IV metrics for tooltip
                        const riskScore = row['risk_score'] !== undefined && row['risk_score'] !== null ? parseFloat(row['risk_score']) : null;
                        const ivRank30 = row['iv_rank_30'] !== undefined && row['iv_rank_30'] !== null ? parseFloat(row['iv_rank_30']) : null;
                        const ivRank90 = row['iv_rank_90'] !== undefined && row['iv_rank_90'] !== null ? parseFloat(row['iv_rank_90']) : null;
                        // Try multiple possible column names (full, truncated, compact)
                        const ivRecommendationRaw = (row['iv_recommendation'] || row['iv_recommendati'] || row['iv_rec'] || null);
                        const ivRecommendation = ivRecommendationRaw ? String(ivRecommendationRaw).trim() : null;
                        const rollYield = row['roll_yield'] !== undefined && row['roll_yield'] !== null ? parseFloat(row['roll_yield']) : null;
                        
                        // Build tooltip text
                        let tooltipText = 'IV Metrics:\n';
                        tooltipText += `Risk Score: ${riskScore !== null ? riskScore.toFixed(1) : 'N/A'}\n`;
                        if (ivRank30 !== null && ivRank90 !== null) {
                            tooltipText += `IV Rank: ${ivRank30.toFixed(1)}/${ivRank90.toFixed(1)}\n`;
                        } else if (ivRank30 !== null) {
                            tooltipText += `IV Rank: ${ivRank30.toFixed(1)}/-\n`;
                        } else {
                            tooltipText += 'IV Rank: N/A\n';
                        }
                        tooltipText += `IV Recommendation: ${ivRecommendation || 'N/A'}\n`;
                        if (rollYield !== null) {
                            let rollYieldValue = rollYield;
                            if (typeof rollYield === 'string' && rollYield.endsWith('%')) {
                                rollYieldValue = parseFloat(rollYield.replace('%', ''));
                            }
                            tooltipText += `Roll Yield: ${rollYieldValue.toFixed(2)}%`;
                        } else {
                            tooltipText += 'Roll Yield: N/A';
                        }
                        
                        tickerLink.title = tooltipText;
                        td.innerHTML = '';
                        td.appendChild(tickerLink);
                    }
                    
                    tr.appendChild(td);
                });
                
                tbody.appendChild(tr);
            });
            
            // Apply all post-render functions (matching original behavior)
            applyRowStriping(prefix);
            applyColumnStriping(prefix);
            syncCardOrder(prefix);
            syncCardVisibility(prefix);
            updateVisibleCount(prefix);
            
            // Initialize column map after data is loaded (needed for filter parsing)
            initColumnMap(prefix);
            
            // Apply sort from URL if it wasn't applied yet (table headers now exist)
            applySortFromURL(prefix);
            
            // Apply client-side filtering for virtual fields (bid, ask, l_bid, l_ask, spread, l_spread)
            // These can't be filtered server-side, so we filter after data loads
            applyVirtualFieldFilters(prefix);
        }
        
        // Show loading indicator
        function showLoadingIndicator(prefix) {
            const loadingEl = document.getElementById(prefix + 'loadingIndicator');
            const tableWrapper = document.getElementById(prefix + 'tableWrapper');
            if (loadingEl) {
                loadingEl.classList.add('active');
            }
            // Hide table wrapper while loading
            if (tableWrapper) {
                tableWrapper.style.opacity = '0.5';
                tableWrapper.style.pointerEvents = 'none';
            }
        }
        
        // Hide loading indicator
        function hideLoadingIndicator(prefix) {
            const loadingEl = document.getElementById(prefix + 'loadingIndicator');
            const tableWrapper = document.getElementById(prefix + 'tableWrapper');
            if (loadingEl) {
                loadingEl.classList.remove('active');
            }
            // Show table wrapper after loading
            if (tableWrapper) {
                tableWrapper.style.opacity = '1';
                tableWrapper.style.pointerEvents = 'auto';
            }
        }
        
        // Namespaced state for calls and puts
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
                const colName = th.textContent.trim().replace(/\s+/g, ' ').replace(/\n/g, '');
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
        
        // Parse bid:ask string and return {bid, ask} or null
        function parseBidAsk(bidAskStr) {
            if (!bidAskStr || typeof bidAskStr !== 'string') return null;
            const parts = bidAskStr.trim().split(':');
            if (parts.length !== 2) return null;
            const bid = parseFloat(parts[0]);
            const ask = parseFloat(parts[1]);
            if (isNaN(bid) || isNaN(ask)) return null;
            return {bid: bid, ask: ask};
        }
        
        // Calculate spread (absolute difference between bid and ask) from bid:ask string
        function calculateSpread(bidAskStr) {
            const parsed = parseBidAsk(bidAskStr);
            if (!parsed) return null;
            return Math.abs(parsed.ask - parsed.bid);
        }
        
        // Extract bid from bid:ask string
        function extractBid(bidAskStr) {
            const parsed = parseBidAsk(bidAskStr);
            return parsed ? parsed.bid : null;
        }
        
        // Extract ask from bid:ask string
        function extractAsk(bidAskStr) {
            const parsed = parseBidAsk(bidAskStr);
            return parsed ? parsed.ask : null;
        }
        
        // Get virtual field value (spread, l_spread, bid, ask, l_bid, l_ask, tkr_info.*) from a row
        function getVirtualFieldValue(fieldName, row, tableId) {
            const lowerField = fieldName.toLowerCase().replace(/\s+/g, '_').trim();
            
            // Helper to get bid:ask string from a column
            function getBidAskString(columnNames) {
                for (const colName of columnNames) {
                    const colIndex = findColumnIndex(colName, tableId);
                    if (colIndex >= 0) {
                        const cell = row.cells[colIndex];
                        if (cell) {
                            return getRawText(cell);
                        }
                    }
                }
                return null;
            }
            
            // Handle tkr_info.* fields - extract from row data directly
            if (lowerField.startsWith('tkr_info.')) {
                const subField = lowerField.substring(9); // Remove 'tkr_info.' prefix
                // Map subfield names to actual row data keys
                const fieldMap = {
                    'risk_score': 'risk_score',
                    'risk': 'risk_score',
                    'iv_rank_30': 'iv_rank_30',
                    'iv_rank30': 'iv_rank_30',
                    'rank_30': 'iv_rank_30',
                    'iv_rank_90': 'iv_rank_90',
                    'iv_rank90': 'iv_rank_90',
                    'rank_90': 'iv_rank_90',
                    'iv_recommendation': 'iv_recommendation',
                    'recommendation': 'iv_recommendation',
                    'rec': 'iv_recommendation',
                    'roll_yield': 'roll_yield',
                    'roll': 'roll_yield',
                    'yield': 'roll_yield'
                };
                const actualField = fieldMap[subField] || subField;
                const value = row[actualField];
                if (value !== undefined && value !== null) {
                    // Parse roll yield if it's a string with %
                    if (actualField === 'roll_yield' && typeof value === 'string' && value.endsWith('%')) {
                        return parseFloat(value.replace('%', ''));
                    }
                    return typeof value === 'number' ? value : parseFloat(value);
                }
                return null;
            }
            
            if (lowerField === 'spread') {
                const bidAskStr = getBidAskString(['bid:ask', 'bid_ask']);
                return bidAskStr ? calculateSpread(bidAskStr) : null;
            }
            
            if (lowerField === 'l_spread' || lowerField === 'long_spread') {
                const bidAskStr = getBidAskString(['l_bid:ask', 'l_bid_ask', 'long_bid_ask']);
                return bidAskStr ? calculateSpread(bidAskStr) : null;
            }
            
            if (lowerField === 'bid') {
                const bidAskStr = getBidAskString(['bid:ask', 'bid_ask']);
                return bidAskStr ? extractBid(bidAskStr) : null;
            }
            
            if (lowerField === 'ask') {
                const bidAskStr = getBidAskString(['bid:ask', 'bid_ask']);
                return bidAskStr ? extractAsk(bidAskStr) : null;
            }
            
            if (lowerField === 'l_bid' || lowerField === 'long_bid') {
                const bidAskStr = getBidAskString(['l_bid:ask', 'l_bid_ask', 'long_bid_ask']);
                return bidAskStr ? extractBid(bidAskStr) : null;
            }
            
            if (lowerField === 'l_ask' || lowerField === 'long_ask') {
                const bidAskStr = getBidAskString(['l_bid:ask', 'l_bid_ask', 'long_bid_ask']);
                return bidAskStr ? extractAsk(bidAskStr) : null;
            }
            
            return null;
        }
        
        // Check if a field name is a virtual field (calculated on-the-fly)
        function isVirtualField(fieldName) {
            if (!fieldName) return false;
            const lowerField = fieldName.toLowerCase().replace(/\s+/g, '_').trim();
            return (
                lowerField === 'spread' ||
                lowerField === 'l_spread' ||
                lowerField === 'long_spread' ||
                lowerField === 'bid' ||
                lowerField === 'ask' ||
                lowerField === 'l_bid' ||
                lowerField === 'long_bid' ||
                lowerField === 'l_ask' ||
                lowerField === 'long_ask' ||
                lowerField.startsWith('tkr_info.')
            );
        }
        
        // Check if a field name is an expiration date column
        function isExpirationDateColumn(fieldName) {
            if (!fieldName) return false;
            const lowerField = fieldName.toLowerCase().replace(/\s+/g, '_').trim();
            return (
                lowerField === 'expiration_date' ||
                lowerField === 'l_expiration_date' ||
                lowerField === 'long_expiration_date' ||
                lowerField === 's_expiration_date' ||
                lowerField === 'short_expiration_date' ||
                lowerField.endsWith('_expiration_date') ||
                lowerField.endsWith('_exp_date')
            );
        }
        
        // Check if a string is a date in YYYY-MM-DD format
        function isDateString(value) {
            if (typeof value !== 'string') return false;
            const datePattern = /^\d{4}-\d{2}-\d{2}$/;
            return datePattern.test(value.trim());
        }
        
        // Extract date portion (YYYY-MM-DD) from a value
        // Handles timestamps (milliseconds), date strings, and other formats
        function extractDatePortion(value) {
            if (value === null || value === undefined || value === '') return null;
            
            // If it's already a date string in YYYY-MM-DD format
            if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value.trim())) {
                return value.trim();
            }
            
            // Try to parse as timestamp (milliseconds)
            const numValue = typeof value === 'string' ? parseFloat(value) : value;
            if (!isNaN(numValue) && numValue > 0) {
                // Check if it's a reasonable timestamp (milliseconds since epoch)
                // Timestamps are typically > 1000000000000 (year 2001) and < 9999999999999 (year 2286)
                if (numValue > 1000000000000 && numValue < 9999999999999) {
                    const date = new Date(numValue);
                    if (!isNaN(date.getTime())) {
                        const year = date.getFullYear();
                        const month = String(date.getMonth() + 1).padStart(2, '0');
                        const day = String(date.getDate()).padStart(2, '0');
                        return `${year}-${month}-${day}`;
                    }
                }
            }
            
            // Try to parse as date string (might have time portion)
            if (typeof value === 'string') {
                const trimmed = value.trim();
                // Extract YYYY-MM-DD from strings like "2026-03-01 00:00:00" or "2026-03-01T00:00:00"
                const dateMatch = trimmed.match(/^(\d{4}-\d{2}-\d{2})/);
                if (dateMatch) {
                    return dateMatch[1];
                }
                // Try parsing as Date object
                const date = new Date(trimmed);
                if (!isNaN(date.getTime())) {
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    return `${year}-${month}-${day}`;
                }
            }
            
            return null;
        }
        
        // Get date value from a cell (for expiration date columns)
        function getDateValue(cell) {
            // First try data-raw attribute
            const rawValue = cell.getAttribute('data-raw');
            if (rawValue !== null && rawValue !== '') {
                const datePortion = extractDatePortion(rawValue);
                if (datePortion) return datePortion;
            }
            
            // Fallback to text content
            const text = cell.textContent.trim();
            if (text) {
                const datePortion = extractDatePortion(text);
                if (datePortion) return datePortion;
            }
            
            return null;
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
                const escapedField = field.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
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
                        columnNames.push(displaySpan.textContent.trim().replace(/\s+/g, ' '));
                    } else {
                        const headerText = th.textContent.trim().replace(/\s+/g, ' ').replace(/\n/g, '');
                        if (headerText) {
                            columnNames.push(headerText);
                        }
                    }
                }
            });
            // Add virtual fields
            columnNames.push('spread', 'l_spread', 'long_spread');
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
            const lowerField = fieldName.toLowerCase().replace(/\s+/g, '_').trim();
            
            if (!lowerField) return -1;
            
            // First, try to match against data-filterable-name attribute (most reliable)
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\s+/g, '_').trim();
                    // Exact match (most reliable)
                    if (normalizedFilterable === lowerField) {
                        return i;
                    }
                }
            }
            
            // Try exact match against header text (normalized)
            for (let i = 0; i < headers.length; i++) {
                const headerText = headers[i].textContent.trim().toLowerCase().replace(/\s+/g, '_').replace(/\n/g, '').trim();
                if (headerText === lowerField) {
                    return i;
                }
            }
            
            // Try matching against data-filterable-name with normalization
            for (let i = 0; i < headers.length; i++) {
                const filterableName = headers[i].getAttribute('data-filterable-name');
                if (filterableName) {
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\s+/g, '_').trim();
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
                    const normalizedFilterable = filterableName.toLowerCase().replace(/\s+/g, '_').trim();
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
            const existsMatch = expression.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s+(exists|not_exists)$/i);
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
                        const hasMath = /[+\-*/]/.test(fieldExpr);
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
                                valueStr: valueStr,  // Preserve original string for percentage detection
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
                        // Check if this is a date string (YYYY-MM-DD format) - preserve as string
                        const isDateStr = isDateString(valueStr);
                        if (!isDateStr) {
                            // Try to parse as number
                            if (fieldExpr.toLowerCase().includes('market_cap') || fieldExpr.toLowerCase().includes('market cap')) {
                                value = parseMarketCapValue(valueStr);
                            } else {
                                const numValue = parseFloat(valueStr);
                                if (!isNaN(numValue)) {
                                    value = numValue;
                                }
                            }
                        }
                        // If it's a date string, keep it as a string (value = valueStr)
                        
                        return {
                            field: fieldExpr,
                            operator: op,
                            value: value,
                            valueStr: valueStr,  // Preserve original string for percentage detection
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
                
                // Check if this is a virtual field
                if (isVirtualField(filter.field)) {
                    const virtualValue = getVirtualFieldValue(filter.field, row, tableId);
                    if (filter.operator === 'exists') {
                        return virtualValue !== null && !isNaN(virtualValue);
                    }
                    if (filter.operator === 'not_exists') {
                        return virtualValue === null || isNaN(virtualValue);
                    }
                }
                
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
                // Check if either field is a virtual field
                const isFieldVirtual = isVirtualField(filter.field);
                const isValueVirtual = isVirtualField(filter.value);
                
                let cellValue = null;
                let compareValue = null;
                
                if (isFieldVirtual) {
                    cellValue = getVirtualFieldValue(filter.field, row, tableId);
                } else {
                    const colIndex = findColumnIndex(filter.field, tableId);
                    if (colIndex < 0) {
                        console.warn('Column not found for filter field:', filter.field);
                        return false;
                    }
                    const cell = row.cells[colIndex];
                    if (!cell) return false;
                    cellValue = getRawValue(cell);
                }
                
                if (isValueVirtual) {
                    compareValue = getVirtualFieldValue(filter.value, row, tableId);
                } else {
                    const valueColIndex = findColumnIndex(filter.value, tableId);
                    if (valueColIndex < 0) {
                        console.warn('Column not found for filter comparison value:', filter.value);
                        return false;
                    }
                    const valueCell = row.cells[valueColIndex];
                    if (!valueCell) return false;
                    compareValue = getRawValue(valueCell);
                }
                
                // Check if both fields are expiration date columns (only if neither is virtual)
                if (!isFieldVirtual && !isValueVirtual) {
                    const isFieldExpDate = isExpirationDateColumn(filter.field);
                    const isValueExpDate = isExpirationDateColumn(filter.value);
                    
                    // Handle date-only comparison for expiration date columns
                    if (isFieldExpDate && isValueExpDate) {
                        const colIndex = findColumnIndex(filter.field, tableId);
                        const valueColIndex = findColumnIndex(filter.value, tableId);
                        const cell = row.cells[colIndex];
                        const valueCell = row.cells[valueColIndex];
                        const cellDate = getDateValue(cell);
                        const compareDate = getDateValue(valueCell);
                        if (cellDate === null || compareDate === null) return false;
                        
                        // Compare date strings directly (YYYY-MM-DD format allows string comparison)
                        switch (filter.operator) {
                            case '>': return cellDate > compareDate;
                            case '>=': return cellDate >= compareDate;
                            case '<': return cellDate < compareDate;
                            case '<=': return cellDate <= compareDate;
                            case '==': return cellDate === compareDate;
                            case '!=': return cellDate !== compareDate;
                            default: return false;
                        }
                    }
                }
                
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
            // Check if this is a virtual field first
            if (isVirtualField(filter.field)) {
                const virtualValue = getVirtualFieldValue(filter.field, row, tableId);
                if (virtualValue === null || isNaN(virtualValue)) return false;
                
                // Handle percentage-based filtering (e.g., "spread < 10%" means spread < 10% of option premium)
                // Use original string value if available, otherwise convert to string
                const filterValueStr = (filter.valueStr !== undefined) ? filter.valueStr.trim() : String(filter.value).trim();
                let filterValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
                
                // Check if the filter value ends with % for percentage-based comparison
                if (filterValueStr.endsWith('%')) {
                    const percentValue = parseFloat(filterValueStr.slice(0, -1));
                    if (!isNaN(percentValue)) {
                        // For spread percentage, we need to compare against option premium
                        // Try to find option_premium column to calculate percentage
                        const lowerField = filter.field.toLowerCase().replace(/\s+/g, '_').trim();
                        if (lowerField === 'spread') {
                            // Find option_premium for short leg - try multiple variations
                            let premColIndex = findColumnIndex('option_premium', tableId);
                            if (premColIndex < 0) {
                                premColIndex = findColumnIndex('opt_prem', tableId);
                            }
                            if (premColIndex < 0) {
                                premColIndex = findColumnIndex('opt_prem.', tableId);
                            }
                            
                            if (premColIndex >= 0) {
                                const premCell = row.cells[premColIndex];
                                if (premCell) {
                                    const premValue = getRawValue(premCell);
                                    if (premValue !== null && !isNaN(premValue) && premValue > 0) {
                                        filterValue = (percentValue / 100) * premValue;
                                    } else {
                                        return false; // Can't calculate percentage without premium
                                    }
                                } else {
                                    return false;
                                }
                            } else {
                                return false; // Can't find premium column
                            }
                        } else if (lowerField === 'l_spread' || lowerField === 'long_spread') {
                            // Find option_premium for long leg - try multiple variations
                            let premColIndex = findColumnIndex('long_option_premium', tableId);
                            if (premColIndex < 0) {
                                premColIndex = findColumnIndex('l_prem', tableId);
                            }
                            if (premColIndex < 0) {
                                premColIndex = findColumnIndex('l_opt_prem', tableId);
                            }
                            
                            if (premColIndex >= 0) {
                                const premCell = row.cells[premColIndex];
                                if (premCell) {
                                    const premValue = getRawValue(premCell);
                                    if (premValue !== null && !isNaN(premValue) && premValue > 0) {
                                        filterValue = (percentValue / 100) * premValue;
                                    } else {
                                        return false; // Can't calculate percentage without premium
                                    }
                                } else {
                                    return false;
                                }
                            } else {
                                return false; // Can't find premium column
                            }
                        } else {
                            // For other virtual fields, just use the percentage as-is (not implemented yet)
                            filterValue = percentValue;
                        }
                    }
                }
                
                // Numeric comparison for virtual field
                if (isNaN(filterValue)) {
                    return false;
                }
                
                switch (filter.operator) {
                    case '>': return virtualValue > filterValue;
                    case '>=': return virtualValue >= filterValue;
                    case '<': return virtualValue < filterValue;
                    case '<=': return virtualValue <= filterValue;
                    case '==': return Math.abs(virtualValue - filterValue) < 0.0001;
                    case '!=': return Math.abs(virtualValue - filterValue) >= 0.0001;
                    default: return false;
                }
            }
            
            const colIndex = findColumnIndex(filter.field, tableId);
            if (colIndex < 0) {
                console.warn('Column not found for filter field:', filter.field);
                return false;
            }
            const cell = row.cells[colIndex];
            if (!cell) return false;
            
            // Check if this is an expiration date column and filter value is a date string
            const isExpDateCol = isExpirationDateColumn(filter.field);
            const filterValueStr = String(filter.value).trim();
            const isDateFilter = isDateString(filterValueStr);
            
            // Handle date-only comparison for expiration date columns
            if (isExpDateCol && isDateFilter) {
                const cellDate = getDateValue(cell);
                if (cellDate === null) return false;
                
                // Compare date strings directly (YYYY-MM-DD format allows string comparison)
                switch (filter.operator) {
                    case '>': return cellDate > filterValueStr;
                    case '>=': return cellDate >= filterValueStr;
                    case '<': return cellDate < filterValueStr;
                    case '<=': return cellDate <= filterValueStr;
                    case '==': return cellDate === filterValueStr;
                    case '!=': return cellDate !== filterValueStr;
                    default: return false;
                }
            }
            
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
            // Use API-based filtering instead of client-side
            // Note: Virtual fields (bid, ask, l_bid, l_ask, spread, l_spread) will be filtered client-side after data loads
            fetchDataFromAPI(prefix);
        }
        
        // Apply client-side filtering for virtual fields after data loads
        function applyVirtualFieldFilters(prefix) {
            if (!activeFilters[prefix] || activeFilters[prefix].length === 0) return;
            
            // Check if there are any virtual field filters
            const hasVirtualFilters = activeFilters[prefix].some(f => isVirtualField(f.field));
            if (!hasVirtualFilters) return; // No virtual field filters, server-side filtering is sufficient
            
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const tbody = table.querySelector('tbody');
            if (!tbody) return;
            
            const rows = Array.from(tbody.querySelectorAll('tr'));
            let visibleCount = 0;
            const logic = filterLogic[prefix] || 'AND';
            
            // Check each row against virtual field filters
            rows.forEach(row => {
                // Get all virtual field filters
                const virtualFilters = activeFilters[prefix].filter(f => isVirtualField(f.field));
                
                if (virtualFilters.length === 0) {
                    // No virtual filters, row is already filtered by server
                    visibleCount++;
                    return;
                }
                
                let passesVirtualFilters = false;
                
                if (logic === 'OR') {
                    // For OR logic: row passes if ANY virtual filter passes
                    passesVirtualFilters = virtualFilters.some(filter => evaluateFilter(filter, row, tableId));
                } else {
                    // For AND logic: row passes if ALL virtual filters pass
                    passesVirtualFilters = virtualFilters.every(filter => evaluateFilter(filter, row, tableId));
                }
                
                // Show/hide row based on virtual filter results
                // Note: Server-side filters are already applied, so we're just applying virtual filters on top
                if (passesVirtualFilters) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });
            
            // Update visible count
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleCount;
            }
            
            // Reapply striping after filtering
            applyRowStriping(prefix);
            syncCardVisibility(prefix);
        }
        
        // Validate filter fields exist
        function validateFilter(filter, tableId) {
            const errorMessages = [];
            
            // Check main field
            if (filter.field && !filter.hasMath) {
                // Check if it's a virtual field first
                if (!isVirtualField(filter.field)) {
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
                }
            } else if (filter.hasMath) {
                // For math expressions, extract field names and validate them
                const fieldPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
                const fields = filter.field.match(fieldPattern) || [];
                const jsKeywords = ['if', 'else', 'for', 'while', 'function', 'return', 'var', 'let', 'const', 'true', 'false', 'null', 'undefined'];
                const validFields = fields.filter(f => !jsKeywords.includes(f.toLowerCase()));
                
                for (const field of validFields) {
                    // Skip validation for virtual fields
                    if (!isVirtualField(field)) {
                        const colIndex = findColumnIndex(field, tableId);
                        if (colIndex < 0) {
                            errorMessages.push(`Field "${field}" in expression "${filter.field}" not found.`);
                        }
                    }
                }
            }
            
            // Check comparison field if it's a field-to-field comparison
            if (filter.isFieldComparison && filter.value) {
                // Check if it's a virtual field first
                if (!isVirtualField(filter.value)) {
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
                // Use valueStr if available (preserves % sign), otherwise use value
                const valueToDisplay = (filter.valueStr !== undefined) ? filter.valueStr : (filter.value !== null ? filter.value : '');
                filterText.textContent = `${filter.field} ${filter.operator} ${valueToDisplay}`;
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
            const cardWrapperId = prefix + 'cardWrapper';
            const cardWrapper = document.getElementById(cardWrapperId);
            
            let visibleCount = 0;
            
            // Check table view (desktop)
            if (table) {
                const tbody = table.querySelector('tbody');
                if (tbody) {
                    const visibleRows = Array.from(tbody.querySelectorAll('tr')).filter(row => row.style.display !== 'none');
                    visibleCount = visibleRows.length;
                }
            }
            
            // Check card view (mobile)
            if (cardWrapper) {
                const cards = Array.from(cardWrapper.querySelectorAll('.data-card'));
                const visibleCards = cards.filter(card => !card.classList.contains('hidden') && card.style.display !== 'none');
                if (visibleCards.length > 0) {
                    visibleCount = visibleCards.length;
                }
            }
            
            // Update visible count element
            const visibleCountEl = document.getElementById(prefix + 'visibleCount');
            if (visibleCountEl) {
                visibleCountEl.textContent = visibleCount;
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
            // Use API-based sorting instead of client-side
            currentSortColumn[prefix] = columnIndex;
            if (!sortDirection[prefix]) sortDirection[prefix] = {};
            if (sortDirection[prefix][columnIndex] === 'asc') {
                sortDirection[prefix][columnIndex] = 'desc';
            } else {
                sortDirection[prefix][columnIndex] = 'asc';
            }
            // Update visual sort indicators
            updateSortIndicators(prefix);
            // Update URL with new sort state
            updateURL();
            fetchDataFromAPI(prefix);
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
        // CRITICAL FIX: Tab switching - only use CSS classes, no inline styles
        function switchTab(tabIndex) {
            // Determine which tab is being switched to and fetch data if needed
            const tabContents = document.querySelectorAll('.tab-content');
            const tabButtons = document.querySelectorAll('.tab-button');
            
            // Hide all tabs
            tabContents.forEach((tab, idx) => {
                tab.classList.remove('active');
                if (tabButtons[idx]) {
                    tabButtons[idx].classList.remove('active');
                }
            });
            
            // Show selected tab
            if (tabContents[tabIndex]) {
                tabContents[tabIndex].classList.add('active');
                if (tabButtons[tabIndex]) {
                    tabButtons[tabIndex].classList.add('active');
                }
                
                // Update URL with tab parameter
                const tabNames = ['calls', 'puts', 'analysis', 'stock_analysis'];
                const tabName = tabNames[tabIndex] || 'calls';
                const params = new URLSearchParams(window.location.search);
                params.set('tab', tabName);
                const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
                window.history.pushState({ tab: tabIndex }, '', newURL);

                // Fetch data for the active tab
                const activeTab = tabContents[tabIndex];
                if (activeTab.id === 'callsTab') {
                    fetchDataFromAPI('calls');
                } else if (activeTab.id === 'putsTab') {
                    fetchDataFromAPI('puts');
                } else if (activeTab.id === 'analysisTab') {
                    // Auto-load rule-based analysis when tab is clicked
                    // User can then click button again with Gemini checkbox checked for AI analysis
                    const useGeminiCheckbox = document.getElementById('useGeminiAnalysis');
                    if (useGeminiCheckbox && !useGeminiCheckbox.checked) {
                        // Only auto-load if Gemini is not checked (rule-based analysis)
                        fetchAnalysisFromAPI();
                    }
                } else if (activeTab.id === 'stockAnalysisTab') {
                    fetchStockAnalysisData();
                }
            }
        }
        
// Get initial tab from URL
function getInitialTab() {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get('tab');
    if (tabParam) {
        const tabMap = {
            'calls': 0,
            'puts': 1,
            'analysis': 2,
            'stock_analysis': 3
        };
        return tabMap[tabParam] !== undefined ? tabMap[tabParam] : 0;
    }
    return 0; // Default to calls tab
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
        
        // Update URL with current filters and sort
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
                            // Use valueStr if available (preserves % sign), otherwise use value
                            const valueToUse = (f.valueStr !== undefined) ? f.valueStr : String(f.value);
                            filterStr += ' ' + valueToUse;
                        }
                    }
                    return filterStr;
                });
                params.set('calls_filters', filterStrings.join('|'));
            }
            
            // Add sort for calls
            if (currentSortColumn['calls'] !== undefined && currentSortColumn['calls'] >= 0) {
                const tableId = 'callsresultsTable';
                const table = document.getElementById(tableId);
                if (table) {
                    const headerRow = table.querySelector('tr.column-header-row');
                    const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
                    if (headers[currentSortColumn['calls']]) {
                        const colName = headers[currentSortColumn['calls']].getAttribute('data-filterable-name') || 
                                      headers[currentSortColumn['calls']].textContent.trim();
                        const sortDir = sortDirection['calls'] && sortDirection['calls'][currentSortColumn['calls']] || 'desc';
                        params.set('calls_sort', colName);
                        params.set('calls_sort_direction', sortDir);
                    }
                }
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
                            // Use valueStr if available (preserves % sign), otherwise use value
                            const valueToUse = (f.valueStr !== undefined) ? f.valueStr : String(f.value);
                            filterStr += ' ' + valueToUse;
                        }
                    }
                    return filterStr;
                });
                params.set('puts_filters', filterStrings.join('|'));
            }
            
            // Add sort for puts
            if (currentSortColumn['puts'] !== undefined && currentSortColumn['puts'] >= 0) {
                const tableId = 'putsresultsTable';
                const table = document.getElementById(tableId);
                if (table) {
                    const headerRow = table.querySelector('tr.column-header-row');
                    const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
                    if (headers[currentSortColumn['puts']]) {
                        const colName = headers[currentSortColumn['puts']].getAttribute('data-filterable-name') || 
                                      headers[currentSortColumn['puts']].textContent.trim();
                        const sortDir = sortDirection['puts'] && sortDirection['puts'][currentSortColumn['puts']] || 'desc';
                        params.set('puts_sort', colName);
                        params.set('puts_sort_direction', sortDir);
                    }
                }
            }
            
            // Update URL without reloading page
            const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
            window.history.replaceState({}, '', newURL);
        }
        
        // Load filters and sort from URL
        function loadFiltersFromURL() {
            const params = new URLSearchParams(window.location.search);
            
            // Load filters for calls
            loadFiltersForPrefix('calls', params);
            
            // Load filters for puts
            loadFiltersForPrefix('puts', params);
            
            // Load sort for calls and puts
            loadSortFromURL();
        }
        
        // Store sort state from URL (to be applied when table is rendered)
        const pendingSortFromURL = {};
        
        // Load sort state from URL
        function loadSortFromURL() {
            const params = new URLSearchParams(window.location.search);
            
            // Store sort for calls (will be applied when table is rendered)
            const callsSortCol = params.get('calls_sort');
            const callsSortDir = params.get('calls_sort_direction');
            if (callsSortCol) {
                pendingSortFromURL['calls'] = {
                    column: callsSortCol,
                    direction: (callsSortDir === 'asc' || callsSortDir === 'desc') ? callsSortDir : 'desc'
                };
            }
            
            // Store sort for puts (will be applied when table is rendered)
            const putsSortCol = params.get('puts_sort');
            const putsSortDir = params.get('puts_sort_direction');
            if (putsSortCol) {
                pendingSortFromURL['puts'] = {
                    column: putsSortCol,
                    direction: (putsSortDir === 'asc' || putsSortDir === 'desc') ? putsSortDir : 'desc'
                };
            }
        }
        
        // Apply sort from URL after table is rendered
        function applySortFromURL(prefix) {
            const pendingSort = pendingSortFromURL[prefix];
            if (!pendingSort) return; // No pending sort for this prefix
            
            // Find the column index by matching the column name
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            for (let i = 0; i < headers.length; i++) {
                const colName = headers[i].getAttribute('data-filterable-name') || headers[i].textContent.trim();
                if (colName === pendingSort.column) {
                    currentSortColumn[prefix] = i;
                    if (!sortDirection[prefix]) sortDirection[prefix] = {};
                    sortDirection[prefix][i] = pendingSort.direction;
                    // Clear pending sort (already applied)
                    delete pendingSortFromURL[prefix];
                    // Update visual sort indicators
                    updateSortIndicators(prefix);
                    break;
                }
            }
        }
        
        // Update visual sort indicators on column headers
        function updateSortIndicators(prefix) {
            const tableId = prefix + 'resultsTable';
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const headerRow = table.querySelector('tr.column-header-row');
            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
            
            headers.forEach((th, index) => {
                // Remove all sort indicator classes
                th.classList.remove('sort-asc', 'sort-desc');
                
                // Add appropriate class if this column is sorted
                if (currentSortColumn[prefix] === index) {
                    const dir = sortDirection[prefix] && sortDirection[prefix][index];
                    if (dir === 'asc') {
                        th.classList.add('sort-asc');
                    } else if (dir === 'desc') {
                        th.classList.add('sort-desc');
                    }
                }
            });
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
                        // Try to parse filter - tableId might not exist yet, but parseFilterExpression can handle that
                        const filter = parseFilterExpression(filterStr.trim(), tableId);
                        if (filter) {
                            // When loading from URL, skip client-side validation since server will validate
                            // Just add the filter - it will be validated when API is called
                            activeFilters[prefix].push(filter);
                        } else {
                            // If parsing failed, log it but continue
                            validationErrors.push(`Could not parse filter: ${filterStr.trim()}`);
                        }
                    }
                }
                
                // Show validation errors if any (but don't block loading)
                if (validationErrors.length > 0 && errorDiv) {
                    errorDiv.textContent = 'Warning: Some filters from URL could not be parsed: ' + validationErrors.join(' ');
                    errorDiv.style.display = 'block';
                    errorDiv.style.color = '#856404';
                    errorDiv.style.backgroundColor = '#fff3cd';
                }
                
                // If filters were loaded from URL, expand the filter section
                // But only if the table exists (might not exist yet on initial load)
                if (activeFilters[prefix].length > 0) {
                    // Delay expanding filter section until after data loads
                    setTimeout(() => {
                        const filterSection = document.getElementById(prefix + 'filterSection');
                        const button = document.getElementById(prefix + 'toggleFilterBtn');
                        const table = document.getElementById(tableId);
                        if (table) {
                            // Only show filterable names on column headers, not group headers
                            const headerRow = table.querySelector('tr.column-header-row');
                            const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
                            
                            if (filterSection && button) {
                                filterSection.classList.add('expanded');
                                headers.forEach(th => {
                                    th.classList.add('showing-filterable');
                                });
                                button.textContent = '✖️ Hide Filter';
                                button.title = 'Hide filter options and show display column names';
                            }
                        }
                    }, 500); // Increased delay to ensure table is rendered
                    
                    // Update filter display immediately
                    updateFilterDisplay(prefix);
                }
            }
        }
        
        // Update data source timestamp display
        function updateDataSourceTimestamp(isoTimestamp) {
            const generatedTimeEl = document.getElementById('generatedTime');
            const dataTimestampEl = document.getElementById('dataTimestamp');
            
            if (!generatedTimeEl || !dataTimestampEl || !isoTimestamp) return;
            
            try {
                // Update the data-generated attribute with the timestamp
                generatedTimeEl.setAttribute('data-generated', isoTimestamp);
                
                // Parse and format the timestamp
                const date = new Date(isoTimestamp);
                const options = {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    timeZoneName: 'short'
                };
                const formattedDate = date.toLocaleString('en-US', options);
                dataTimestampEl.textContent = formattedDate;
                
                // Immediately update the "time ago" display
                updateTimeAgo();
            } catch (e) {
                console.error('Error updating data source timestamp:', e);
                dataTimestampEl.textContent = 'Unknown';
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
                timeAgoEl.textContent = '';
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
        
// Fetch and render stock analysis data
async function fetchStockAnalysisData() {
    const loadingIndicator = document.getElementById('stockAnalysisLoadingIndicator');
    const contentDiv = document.getElementById('stockAnalysisContent');

    if (loadingIndicator) {
        loadingIndicator.classList.add('active');
    }
    if (contentDiv) {
        contentDiv.innerHTML = '';
    }

    try {
        const url = '/stock_info/api/stock_analysis/data';
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success) {
            renderStockAnalysis(data.data);
        } else {
            throw new Error(data.message || 'Unknown error');
        }
    } catch (error) {
        console.error('Error fetching stock analysis data:', error);
        if (contentDiv) {
            contentDiv.innerHTML = `<div style="padding: 20px; color: red; text-align: center;">Error loading stock analysis: ${error.message}</div>`;
        }
    } finally {
        if (loadingIndicator) {
            loadingIndicator.classList.remove('active');
        }
    }
}

// Render stock analysis data
function renderStockAnalysis(data) {
    const contentDiv = document.getElementById('stockAnalysisContent');
    if (!contentDiv) return;

    let html = '';

    // Total tickers analyzed
    html += `<div style="margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">`;
    html += `<h2 style="color: #667eea; margin-bottom: 10px;">✅ ANALYZED ${data.total_tickers} TICKERS</h2>`;
    html += `</div>`;

    // Strategy sections
    const strategyConfig = [
        { key: 'BACKWARDATION', emoji: '⚠️', name: 'BACKWARDATION' },
        { key: 'WHALE SQUEEZE', emoji: '🐳', name: 'WHALE SQUEEZE' },
        { key: 'SECTOR RELATIVE', emoji: '📊', name: 'SECTOR RELATIVE' },
        { key: 'CASH FLOW KING', emoji: '👑', name: 'CASH FLOW KINGS' },
        { key: 'MEAN REVERSION', emoji: '📈', name: 'MEAN REVERSION' },
        { key: 'ACCUMULATION', emoji: '🟢', name: 'ACCUMULATION' }
    ];

    for (const config of strategyConfig) {
        const strategyData = data.strategies[config.key] || [];
        if (strategyData.length === 0) continue;

        html += `<div style="margin-bottom: 30px;">`;
        html += `<h3 style="color: #667eea; font-size: 1.5em; margin-bottom: 15px;">${config.emoji} ${config.name} (Top ${strategyData.length})</h3>`;
        html += `<table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">`;
        html += `<thead style="background: #667eea; color: white;">`;
        html += `<tr>`;
        html += `<th style="padding: 12px; text-align: left;">Ticker</th>`;
        html += `<th style="padding: 12px; text-align: right;">Price</th>`;
        html += `<th style="padding: 12px; text-align: right;">IV Rank</th>`;
        html += `<th style="padding: 12px; text-align: left;">Sector</th>`;
        html += `</tr>`;
        html += `</thead>`;
        html += `<tbody>`;

        for (const row of strategyData) {
            const ticker = row.ticker || 'N/A';
            const tickerLink = ticker !== 'N/A' ? `<a href="/stock_info/${ticker}" style="color: #667eea; text-decoration: none; font-weight: 600;">${ticker}</a>` : ticker;
            html += `<tr style="border-bottom: 1px solid #dee2e6;">`;
            html += `<td style="padding: 10px 12px; font-weight: 600;">${tickerLink}</td>`;
            html += `<td style="padding: 10px 12px; text-align: right;">$${(row.price || 0).toFixed(2)}</td>`;
            html += `<td style="padding: 10px 12px; text-align: right;">${(row.iv_rank || 0).toFixed(1)}</td>`;
            html += `<td style="padding: 10px 12px;">${row.sector || 'Unknown'}</td>`;
            html += `</tr>`;
        }

        html += `</tbody>`;
        html += `</table>`;
        html += `</div>`;
    }

    // Final ranked opportunities
    if (data.final_ranked && data.final_ranked.length > 0) {
        html += `<div style="margin-top: 40px; margin-bottom: 30px;">`;
        html += `<h2 style="color: #667eea; font-size: 2em; margin-bottom: 20px; text-align: center; border-bottom: 3px solid #667eea; padding-bottom: 10px;">🏆 FINAL RANKED OPPORTUNITIES (CONVICTION RANK)</h2>`;
        html += `<table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">`;
        html += `<thead style="background: #667eea; color: white;">`;
        html += `<tr>`;
        html += `<th style="padding: 12px; text-align: left;">Ticker</th>`;
        html += `<th style="padding: 12px; text-align: center;">Score</th>`;
        html += `<th style="padding: 12px; text-align: right;">Price</th>`;
        html += `<th style="padding: 12px; text-align: left;">Strategies</th>`;
        html += `<th style="padding: 12px; text-align: left;">Action Plan</th>`;
        html += `</tr>`;
        html += `</thead>`;
        html += `<tbody>`;

        for (const row of data.final_ranked) {
            const ticker = row.ticker || 'N/A';
            const tickerLink = ticker !== 'N/A' ? `<a href="/stock_info/${ticker}" style="color: #667eea; text-decoration: none; font-weight: 600;">${ticker}</a>` : ticker;
            html += `<tr style="border-bottom: 1px solid #dee2e6;">`;
            html += `<td style="padding: 10px 12px; font-weight: 600;">${tickerLink}</td>`;
            html += `<td style="padding: 10px 12px; text-align: center;">${row.conviction_score || 0}</td>`;
            html += `<td style="padding: 10px 12px; text-align: right;">$${(row.price || 0).toFixed(2)}</td>`;
            html += `<td style="padding: 10px 12px;">${row.strategies || 'N/A'}</td>`;
            html += `<td style="padding: 10px 12px;">${row.action_plan || 'N/A'}</td>`;
            html += `</tr>`;
        }

        html += `</tbody>`;
        html += `</table>`;
        html += `</div>`;
    }

    // Full CSV table
    if (data.all_data && data.all_data.length > 0) {
        html += `<div style="margin-top: 40px; margin-bottom: 30px;">`;
        html += `<h2 style="color: #667eea; font-size: 2em; margin-bottom: 10px; text-align: center; border-bottom: 3px solid #667eea; padding-bottom: 10px;">📊 Full Analysis Results</h2>`;

        // Filter to only show tickers with at least one strategy
        const filteredData = data.all_data.filter(row => {
            const strategies = row.strategies || '';
            return strategies.trim().length > 0;
        });

        if (filteredData.length === 0) {
            html += `<div style="text-align: center; padding: 40px; color: #666;">No tickers with active strategies found.</div>`;
            html += `</div>`;
            contentDiv.innerHTML = html;
            return;
        }

        // Define strategy types
        const strategyTypes = ['BACKWARDATION', 'WHALE SQUEEZE', 'SECTOR RELATIVE', 'CASH FLOW KING', 'MEAN REVERSION', 'ACCUMULATION'];

        // Process data to extract individual strategies and count them
        const processedData = filteredData.map(row => {
            const newRow = { ...row };
            // Parse strategies string and create individual columns
            const strategiesStr = row.strategies || '';
            let strategyCount = 0;
            strategyTypes.forEach(strategy => {
                const hasStrategy = strategiesStr.includes(strategy);
                newRow['strategy_' + strategy] = hasStrategy ? '✓' : '';
                if (hasStrategy) strategyCount++;
            });
            // Add strategy count for sorting
            newRow._strategyCount = strategyCount;
            return newRow;
        });

        // Sort by number of strategies (descending), then by ticker name
        processedData.sort((a, b) => {
            if (b._strategyCount !== a._strategyCount) {
                return b._strategyCount - a._strategyCount;
            }
            // If same number of strategies, sort by ticker alphabetically
            const tickerA = (a.ticker || '').toUpperCase();
            const tickerB = (b.ticker || '').toUpperCase();
            return tickerA.localeCompare(tickerB);
        });

        // Add count of entries being shown
        html += `<div style="text-align: center; margin-bottom: 20px; color: #666; font-size: 0.9em;">Showing ${processedData.length} ticker${processedData.length !== 1 ? 's' : ''} with active strategies</div>`;

        // Get all unique column names from the processed data
        const allColumns = new Set();
        processedData.forEach(row => {
            Object.keys(row).forEach(key => {
                // Skip internal sorting field
                if (key !== '_strategyCount') {
                    allColumns.add(key);
                }
            });
        });

        // Remove 'strategies' from columns and add individual strategy columns
        allColumns.delete('strategies');
        strategyTypes.forEach(strategy => allColumns.add('strategy_' + strategy));

        // Reorder columns: ticker first, then other columns (excluding strategies), then strategy columns
        const columns = Array.from(allColumns);
        const tickerIndex = columns.indexOf('ticker');
        if (tickerIndex > -1) {
            columns.splice(tickerIndex, 1);
        }
        // Sort non-strategy columns (excluding ticker)
        const nonStrategyCols = columns.filter(col => !col.startsWith('strategy_')).sort();
        const strategyCols = columns.filter(col => col.startsWith('strategy_')).sort();
        // Final order: ticker, then other columns, then strategy columns
        const finalColumns = ['ticker', ...nonStrategyCols, ...strategyCols];

        html += `<div style="overflow-x: auto;">`;
        html += `<table id="stockAnalysisTable" style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.9em;">`;
        html += `<thead style="background: #667eea; color: white; position: sticky; top: 0;">`;
        html += `<tr>`;
        for (let i = 0; i < finalColumns.length; i++) {
            const col = finalColumns[i];
            const isStrategy = col.startsWith('strategy_');
            const displayName = isStrategy ? col.replace('strategy_', '') : col;
            html += `<th style="padding: 10px 8px; text-align: ${isStrategy ? 'center' : 'left'}; white-space: nowrap; cursor: pointer; user-select: none;" onclick="sortStockAnalysisTable(${i})" data-col-index="${i}" data-col-name="${col}">`;
            html += `${displayName} <span class="sort-indicator" style="opacity: 0.5;">↕</span>`;
            html += `</th>`;
        }
        html += `</tr>`;
        html += `</thead>`;
        html += `<tbody id="stockAnalysisTableBody">`;

        for (const row of processedData) {
            html += `<tr style="border-bottom: 1px solid #dee2e6;">`;
            for (const col of finalColumns) {
                const value = row[col];
                let displayValue = value;
                const isStrategy = col.startsWith('strategy_');
                const isTicker = col === 'ticker';

                if (value === null || value === undefined) {
                    displayValue = '';
                } else if (typeof value === 'number') {
                    // Format numbers nicely
                    if (Math.abs(value) >= 1000) {
                        displayValue = value.toFixed(2);
                    } else if (Math.abs(value) >= 1) {
                        displayValue = value.toFixed(2);
                    } else {
                        displayValue = value.toFixed(4);
                    }
                } else {
                    displayValue = String(value);
                }

                // Make ticker a clickable link
                if (isTicker && displayValue) {
                    const tickerValue = String(value);
                    displayValue = `<a href="/stock_info/${tickerValue}" style="color: #667eea; text-decoration: none; font-weight: 600;">${tickerValue}</a>`;
                }

                html += `<td style="padding: 8px; white-space: nowrap; text-align: ${isStrategy ? 'center' : 'left'};" data-value="${value !== null && value !== undefined ? (typeof value === 'number' ? value : String(value).toLowerCase()) : ''}">${displayValue}</td>`;
            }
            html += `</tr>`;
        }

        html += `</tbody>`;
        html += `</table>`;
        html += `</div>`;
        html += `</div>`;

        // Store processed data for sorting
        window.stockAnalysisData = processedData;
        window.stockAnalysisColumns = finalColumns;
    }

    contentDiv.innerHTML = html;
}

// Sort function for stock analysis table
function sortStockAnalysisTable(columnIndex) {
    const table = document.getElementById('stockAnalysisTable');
    const tbody = document.getElementById('stockAnalysisTableBody');
    if (!table || !tbody || !window.stockAnalysisData || !window.stockAnalysisColumns) return;

    const columnName = window.stockAnalysisColumns[columnIndex];
    const headers = table.querySelectorAll('th');
    const currentHeader = headers[columnIndex];

    // Get current sort direction
    let sortDirection = 'asc';
    const sortIndicator = currentHeader.querySelector('.sort-indicator');

    // Check if already sorted (has sort class)
    if (currentHeader.classList.contains('sort-asc')) {
        sortDirection = 'desc';
    } else if (currentHeader.classList.contains('sort-desc')) {
        sortDirection = 'asc';
    }

    // Remove sort classes from all headers
    headers.forEach((th, idx) => {
        th.classList.remove('sort-asc', 'sort-desc');
        const indicator = th.querySelector('.sort-indicator');
        if (indicator) {
            indicator.textContent = '↕';
            indicator.style.opacity = '0.5';
        }
    });

    // Add sort class to current header
    currentHeader.classList.add('sort-' + sortDirection);
    if (sortIndicator) {
        sortIndicator.textContent = sortDirection === 'asc' ? '↑' : '↓';
        sortIndicator.style.opacity = '1';
    }

    // Sort the data
    const sortedData = [...window.stockAnalysisData].sort((a, b) => {
        let aVal = a[columnName];
        let bVal = b[columnName];

        // Handle null/undefined
        if (aVal === null || aVal === undefined) aVal = '';
        if (bVal === null || bVal === undefined) bVal = '';

        // Handle strategy columns (checkmark)
        if (columnName.startsWith('strategy_')) {
            const aHas = aVal === '✓' || aVal === true || aVal === 1;
            const bHas = bVal === '✓' || bVal === true || bVal === 1;
            if (aHas && !bHas) return sortDirection === 'asc' ? -1 : 1;
            if (!aHas && bHas) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        }

        // Handle numbers
        if (typeof aVal === 'number' && typeof bVal === 'number') {
            return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
        }

        // Handle strings
        const aStr = String(aVal).toLowerCase();
        const bStr = String(bVal).toLowerCase();
        if (aStr < bStr) return sortDirection === 'asc' ? -1 : 1;
        if (aStr > bStr) return sortDirection === 'asc' ? 1 : -1;
        return 0;
    });

    // Update the table body
    tbody.innerHTML = '';
    const columns = window.stockAnalysisColumns;
    for (const row of sortedData) {
        let rowHtml = '<tr style="border-bottom: 1px solid #dee2e6;">';
        for (const col of columns) {
            const value = row[col];
            let displayValue = value;
            const isStrategy = col.startsWith('strategy_');
            const isTicker = col === 'ticker';

            if (value === null || value === undefined) {
                displayValue = '';
            } else if (typeof value === 'number') {
                // Format numbers nicely
                if (Math.abs(value) >= 1000) {
                    displayValue = value.toFixed(2);
                } else if (Math.abs(value) >= 1) {
                    displayValue = value.toFixed(2);
                } else {
                    displayValue = value.toFixed(4);
                }
            } else {
                displayValue = String(value);
            }

            // Make ticker a clickable link
            if (isTicker && displayValue) {
                const tickerValue = String(value);
                displayValue = `<a href="/stock_info/${tickerValue}" style="color: #667eea; text-decoration: none; font-weight: 600;">${tickerValue}</a>`;
            }

            rowHtml += `<td style="padding: 8px; white-space: nowrap; text-align: ${isStrategy ? 'center' : 'left'};" data-value="${value !== null && value !== undefined ? (typeof value === 'number' ? value : String(value).toLowerCase()) : ''}">${displayValue}</td>`;
        }
        rowHtml += '</tr>';
        tbody.innerHTML += rowHtml;
    }

    // Update stored data
    window.stockAnalysisData = sortedData;
}

            // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize state for both prefixes
            initStateForPrefix('calls');
            initStateForPrefix('puts');
            
            // Load filters from URL first (before fetching data)
            loadFiltersFromURL();
            
            // Get initial tab from URL, default to 0 (calls)
            const initialTab = getInitialTab();

            // Fetch data from API for active tab
            switchTab(initialTab);
            
            // Update filter displays
            updateFilterDisplay('calls');
            updateFilterDisplay('puts');
            
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
                
                // Default sort by net_daily_premi descending (will be applied via API)
                // But only if no sort is specified in URL
                // Set initial sort state (only if URL doesn't have sort)
                if (!pendingSortFromURL[prefix]) {
                    const tableId = prefix + 'resultsTable';
                    const table = document.getElementById(tableId);
                    if (table) {
                        const headerRow = table.querySelector('tr.column-header-row');
                        const headers = headerRow ? headerRow.querySelectorAll('th') : table.querySelectorAll('th');
                        headers.forEach((th, idx) => {
                            const colName = th.getAttribute('data-filterable-name') || th.textContent.trim();
                            if (colName.toLowerCase().includes('net_daily_premi') || colName.toLowerCase().includes('net_daily_premium')) {
                                currentSortColumn[prefix] = idx;
                                if (!sortDirection[prefix]) sortDirection[prefix] = {};
                                sortDirection[prefix][idx] = 'desc';
                                // Add sort class to header
                                th.classList.add('sort-desc');
                            }
                        });
                    }
                }
            });
        });
