"""
CSS styles generator for HTML report.
"""


def get_css_styles():
    """Get CSS styles for the HTML report.
    
    Returns:
        String containing CSS styles
    """
    return """        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .tabs {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab-button {
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            user-select: none;
        }
        
        .tab-button:hover {
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
        }
        
        .tab-button.active {
            background: white;
            color: #667eea;
            border-color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block !important;
        }
        
        .table-wrapper {
            overflow-x: auto;
            padding: 20px;
            display: block;
        }
        
        /* Hidden columns handling */
        .table-wrapper.hide-hidden th.is-hidden-col,
        .table-wrapper.hide-hidden td.is-hidden-col {
            display: none;
        }
        
        /* Always hidden columns (hidden in all cases) */
        th.always-hidden,
        td.always-hidden {
            display: none !important;
        }
        
        /* Hide group header row when hidden columns are shown */
        .table-wrapper:not(.hide-hidden) tr.group-header-row {
            display: none;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            table-layout: auto;
        }
        
        thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 15px 8px;
            text-align: center;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
            cursor: pointer;
            user-select: none;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
            width: auto;
            max-width: 12ch;
            min-width: fit-content;
            line-height: 1.3;
            position: relative;
        }
        
        /* Group header row styling */
        tr.group-header-row th.group-header {
            background: #667eea;
            color: white;
            text-align: center;
            font-weight: 700;
            font-size: 0.95em;
            border-bottom: 2px solid #5568d3;
            cursor: default;
            padding: 10px 12px;
        }
        
        tr.group-header-row th.group-header.is-hidden-col {
            display: none;
        }
        
        tr.group-header-row th.group-header:hover {
            background: #5568d3;
        }
        
        /* Column header row styling (second row) */
        tr.column-header-row th {
            border-top: 1px solid #dee2e6;
            padding: 10px 12px;
            font-size: 0.9em;
        }
        
        th.even-col {
            background: #f0f0f0;
        }
        
        th:hover {
            background: #e9ecef;
        }
        
        th.even-col:hover {
            background: #d0d0d0;
        }
        
        
        .column-name-display {
            display: block;
        }
        
        .column-name-short-long {
            display: none;
            font-weight: 600;
            font-size: 0.85em;
            color: #495057;
        }
        
        .column-name-filterable {
            display: none;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #667eea;
        }
        
        /* When filter section is NOT expanded, show short/long labels for grouped columns */
        th.grouped-column:not(.showing-filterable) .column-name-display {
            display: none;
        }
        
        th.grouped-column:not(.showing-filterable) .column-name-short-long {
            display: block;
        }
        
        th.showing-filterable .column-name-display {
            display: none;
        }
        
        th.showing-filterable .column-name-short-long {
            display: none;
        }
        
        th.showing-filterable .column-name-filterable {
            display: block;
        }
        
        th.sortable::after {
            content: ' ↕';
            opacity: 0.5;
            font-size: 0.8em;
        }
        
        th.sort-asc::after {
            content: ' ↑';
            opacity: 1;
            color: #667eea;
        }
        
        th.sort-desc::after {
            content: ' ↓';
            opacity: 1;
            color: #667eea;
        }
        
        td {
            padding: 12px 8px;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
            text-align: center;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
            width: auto;
            max-width: 12ch;
            min-width: fit-content;
        }
        
        /* Wrap text if content exceeds 12 characters */
        td, th {
            overflow-wrap: break-word;
            word-break: break-word;
        }
        
        td.even-col {
            background: #f9f9f9;
        }
        
        td .cell-main {
            font-weight: 600;
            line-height: 1.2;
        }
        
        td .cell-sub {
            font-size: 0.8em;
            color: #6c757d;
            line-height: 1.1;
            margin-top: 3px;
        }
        
        td .cell-sub.theta-percent {
            font-weight: 500;
            color: #007bff;
        }
        
        td.price-positive {
            color: #28a745;
            font-weight: 600;
        }
        
        td.price-negative {
            color: #dc3545;
            font-weight: 600;
        }
        
        tbody tr.even-row {
            background: #f8f9fa;
        }
        
        tbody tr.even-row td.even-col {
            background: #f0f0f0;
        }
        
        tbody tr:hover {
            background: #e9ecef !important;
        }
        
        tbody tr:hover td {
            background: #e9ecef !important;
        }
        
        tbody tr.even-row:hover {
            background: #d9d9d9 !important;
        }
        
        tbody tr.even-row:hover td {
            background: #d9d9d9 !important;
        }
        
        .stats {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        .detailed-analysis {
            padding: 30px;
            background: #f8f9fa;
            border-top: 2px solid #dee2e6;
        }
        
        .detailed-analysis h2 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .analysis-item {
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .analysis-item h3 {
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .analysis-section {
            margin-bottom: 20px;
        }
        
        .analysis-section h4 {
            color: #495057;
            font-size: 1.2em;
            margin-bottom: 10px;
            margin-top: 15px;
        }
        
        .analysis-section p {
            margin: 8px 0;
            line-height: 1.6;
            color: #495057;
        }
        
        .analysis-section .label {
            font-weight: 600;
            color: #333;
        }
        
        .option-tickers {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
        }
        
        .option-tickers .short {
            color: #dc3545;
        }
        
        .option-tickers .long {
            color: #28a745;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .risk-low {
            background: #d4edda;
            color: #155724;
        }
        
        .risk-moderate {
            background: #fff3cd;
            color: #856404;
        }
        
        .risk-high {
            background: #f8d7da;
            color: #721c24;
        }
        
        .score-badge {
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .filter-section {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 20px;
            display: none; /* Hidden by default */
        }
        
        .filter-section.expanded {
            display: block;
        }
        
        .filter-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .filter-input-group {
            display: flex;
            align-items: center;
            gap: 5px;
            flex: 1;
            min-width: 250px;
        }
        
        .filter-input {
            flex: 1;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }
        
        .filter-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .filter-button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        
        .filter-button:hover {
            background: #5568d3;
        }
        
        .filter-button.clear {
            background: #6c757d;
        }
        
        .filter-button.clear:hover {
            background: #5a6268;
        }
        
        .filter-logic {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .filter-logic label {
            font-weight: 600;
            color: #495057;
        }
        
        .filter-logic input[type="radio"] {
            margin-right: 5px;
        }
        
        .filter-help {
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }
        
        .filter-help code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        /* Loading indicator */
        .loading-indicator {
            display: none;
            position: relative;
            text-align: center;
            padding: 40px 20px;
            background: white;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .loading-indicator.active {
            display: block;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .pulse-dots {
            display: inline-block;
        }
        
        .pulse-dots span {
            display: inline-block;
            animation: pulse 1.4s ease-in-out infinite;
            margin: 0 2px;
            font-size: 20px;
            color: #667eea;
        }
        
        .pulse-dots span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .pulse-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .pulse-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 0.3;
                transform: scale(1);
            }
            50% {
                opacity: 1;
                transform: scale(1.2);
            }
        }
        
        .loading-text {
            color: #667eea;
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 10px;
        }
        
        /* Mobile loading indicator */
        @media (max-width: 768px) {
            .loading-indicator {
                padding: 30px 15px;
                margin: 15px 0;
            }
            
            .loading-spinner {
                width: 40px;
                height: 40px;
                border-width: 3px;
            }
            
            .loading-text {
                font-size: 1em;
            }
        }
        
        .filter-error {
            color: #dc3545;
            font-size: 0.9em;
            margin-top: 5px;
            padding: 5px;
            background: #f8d7da;
            border-radius: 3px;
        }
        
        /* Mobile Card Layout */
        .card-wrapper {
            display: none;
            padding: 15px;
        }
        
        .data-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: box-shadow 0.3s ease;
        }
        
        .data-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .data-card.hidden {
            display: none;
        }
        
        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .card-header-main {
            flex: 1;
            min-width: 200px;
        }
        
        .card-ticker {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .card-price {
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .card-price-value {
            font-weight: 600;
        }
        
        .card-price-change {
            font-size: 0.9em;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.2);
        }
        
        .card-price-change.positive {
            background: rgba(40, 167, 69, 0.3);
        }
        
        .card-price-change.negative {
            background: rgba(220, 53, 69, 0.3);
        }
        
        .card-body {
            padding: 15px;
        }
        
        .card-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
            align-items: center;
        }
        
        .card-row:last-child {
            border-bottom: none;
        }
        
        .card-label {
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
            flex: 1;
        }
        
        .card-value {
            color: #212529;
            text-align: right;
            flex: 1;
            font-size: 0.95em;
        }
        
        .card-section {
            margin-bottom: 15px;
        }
        
        .card-section-title {
            font-weight: 700;
            color: #667eea;
            font-size: 0.95em;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #667eea;
        }
        
        .card-details {
            display: none;
        }
        
        .card-details.expanded {
            display: block;
        }
        
        .card-toggle {
            background: #f8f9fa;
            border: none;
            width: 100%;
            padding: 12px;
            cursor: pointer;
            font-size: 0.9em;
            color: #667eea;
            font-weight: 600;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            transition: background 0.3s ease;
        }
        
        .card-toggle:hover {
            background: #e9ecef;
        }
        
        .card-toggle-icon {
            transition: transform 0.3s ease;
        }
        
        .card-toggle.expanded .card-toggle-icon {
            transform: rotate(180deg);
        }
        
        .card-primary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .card-primary-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .card-primary-label {
            font-size: 0.75em;
            color: #6c757d;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .card-primary-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #212529;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            /* Hide table on mobile */
            .table-wrapper {
                display: none !important;
            }
            
            /* Show cards on mobile */
            .card-wrapper {
                display: block !important;
            }
            
            table {
                font-size: 0.8em;
            }
            
            th, td {
                padding: 8px 6px;
            }
            
            .filter-controls {
                flex-direction: column;
            }
            
            .filter-input-group {
                width: 100%;
            }
            
            .container {
                max-width: 100%;
                border-radius: 0;
            }
            
            body {
                padding: 0;
            }
            
            .card-header {
                padding: 12px;
            }
            
            .card-ticker {
                font-size: 1.2em;
            }
            
            .card-body {
                padding: 12px;
            }
        }
        
        @media (min-width: 769px) {
            /* Hide cards on desktop */
            .card-wrapper {
                display: none !important;
            }
            
            /* Show table on desktop */
            .table-wrapper {
                display: block !important;
            }
            
            /* Show desktop-only messages */
            .desktop-only {
                display: block;
            }
            
            .mobile-only {
                display: none;
            }
        }
        
        .desktop-only {
            display: block;
        }
        
        .mobile-only {
            display: none;
        }
        
        /* CRITICAL FIX: Ensure proper tab and view visibility */
        /* Hide inactive tabs completely */
        .tab-content:not(.active) {
            display: none !important;
        }
        
        /* Desktop: Show tables, hide cards */
        @media (min-width: 769px) {
            .tab-content.active .table-wrapper {
                display: block !important;
            }
            .tab-content.active .card-wrapper {
                display: none !important;
            }
            .desktop-only {
                display: block;
            }
            .mobile-only {
                display: none;
            }
        }
        
        /* Mobile: Show cards, hide tables */
        @media (max-width: 768px) {
            .tab-content.active .table-wrapper {
                display: none !important;
            }
            .tab-content.active .card-wrapper {
                display: block !important;
            }
            .desktop-only {
                display: none;
            }
            .mobile-only {
                display: block;
            }
        }
"""
