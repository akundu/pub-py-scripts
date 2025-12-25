// Stock Info Renderer
// This script renders dynamic data into the static template

(function () {
  'use strict';

  // Get JSON data from embedded script tag
  const dataScript = document.getElementById('stockData');
  if (!dataScript || !dataScript.textContent) {
    console.error('No stock data found');
    return;
  }

  let stockData;
  try {
    stockData = JSON.parse(dataScript.textContent);
  } catch (e) {
    console.error('Failed to parse stock data:', e);
    return;
  }

  const symbol = stockData.symbol;
  const priceInfo = stockData.price_info || {};
  const financialInfo = stockData.financial_info || {};
  const optionsInfo = stockData.options_info || {};
  const ivInfo = stockData.iv_info || {};
  const newsInfo = stockData.news_info || {};

  // Formatting utilities
  function formatValue(val, isCurrency = false, isPercentage = false) {
    if (val === null || val === undefined || (typeof val === 'number' && isNaN(val))) {
      return 'N/A';
    }
    if (typeof val === 'number') {
      if (isPercentage) {
        return (val * 100).toFixed(2) + '%';
      } else if (isCurrency) {
        if (val >= 1e9) return '$' + (val / 1e9).toFixed(2) + 'B';
        if (val >= 1e6) return '$' + (val / 1e6).toFixed(2) + 'M';
        if (val >= 1e3) return '$' + (val / 1e3).toFixed(2) + 'K';
        return '$' + val.toFixed(2);
      } else {
        if (Math.abs(val) >= 1e9) return (val / 1e9).toFixed(2) + 'B';
        if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(2) + 'M';
        if (Math.abs(val) >= 1e3) return (val / 1e3).toFixed(2) + 'K';
        return val.toFixed(2);
      }
    }
    return String(val);
  }

  function createMetricCard(label, value) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
        `;
    return card;
  }

  // Render header
  document.getElementById('symbolHeader').textContent = symbol;
  document.getElementById('pageTitle').textContent = `${symbol} - Stock Information`;

  // Render price data
  const currentPriceData = priceInfo.current_price || {};
  const previousClose = currentPriceData.previous_close || financialInfo.financial_data?.previous_close;
  const mostRecentClose = currentPriceData.most_recent_close;

  // Determine if market is open or closed
  const now = new Date();
  const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const dayOfWeek = et.getDay(); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday
  const hours = et.getHours();
  const minutes = et.getMinutes();
  const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
  const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
  const isMarketHours = isWeekday && ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16));
  const isMarketClosed = isWeekend || (isWeekday && (hours < 9 || (hours === 9 && minutes < 30) || hours >= 16));

  // Determine current price to display:
  // - When market is open: use current price
  // - When market is closed: use most_recent_close (set by backend from daily data) or price field
  let currentPrice;
  if (isMarketHours) {
    // Market is open: use current price
    currentPrice = currentPriceData.price || currentPriceData.close || currentPriceData.last_price;
  } else {
    // Market is closed: backend sets 'price' field to most_recent_close from daily data
    // Use that, or fall back to most_recent_close field, or price/close fields
    currentPrice = currentPriceData.price ||
      currentPriceData.most_recent_close ||
      currentPriceData.close ||
      currentPriceData.last_price ||
      previousClose;
    console.log(`[${symbol}] Market CLOSED - currentPrice determined:`, {
      price: currentPriceData.price,
      most_recent_close: currentPriceData.most_recent_close,
      close: currentPriceData.close,
      last_price: currentPriceData.last_price,
      previousClose: previousClose,
      final: currentPrice
    });
  }

  // Backend strategy:
  // - When market is closed: Fetches last 5 days of daily data, sets:
  //   * price = most recent close (359.95) - this is what we display
  //   * previous_close = previous close from daily data (407.4399) - this is for the "Previous Close" metric
  //   * most_recent_trading_day_close = previous close (407.4399) - this is for diff calculation
  // - When market is open: Uses realtime data, previous_close as reference
  // Use most_recent_trading_day_close if available, otherwise fall back to previousClose
  const mostRecentTradingDayClose = currentPriceData.most_recent_trading_day_close ||
    currentPriceData.day_before_last_trading_day_close ||
    previousClose;

  console.log(`[${symbol}] Price diff calculation setup:`, {
    currentPrice: currentPrice,
    previousClose: previousClose,
    most_recent_trading_day_close: currentPriceData.most_recent_trading_day_close,
    day_before_last_trading_day_close: currentPriceData.day_before_last_trading_day_close,
    mostRecentTradingDayClose: mostRecentTradingDayClose,
    isMarketClosed: isMarketClosed
  });

  const priceChange = currentPriceData.change || currentPriceData.change_amount || 0;
  const priceChangePct = currentPriceData.change_percent || currentPriceData.change_pct || 0;

  console.log(`[${symbol}] Raw price fields:`, {
    isMarketHours,
    isMarketClosed,
    price: currentPriceData.price,
    close: currentPriceData.close,
    last_price: currentPriceData.last_price,
    most_recent_close: currentPriceData.most_recent_close,
    previous_close: previousClose,
    most_recent_trading_day_close: currentPriceData.most_recent_trading_day_close,
    day_before_last_trading_day_close: currentPriceData.day_before_last_trading_day_close,
    currentPrice,
    mostRecentTradingDayClose
  });

  // Calculate change for main display using most_recent_trading_day_close
  // Backend sets this to 2 days ago during pre-market, yesterday when market opens
  // On weekends: backend sets this to day-before-last-trading-day (Thursday if Friday was last)
  let mainPriceChange = 0;
  let mainPriceChangePct = 0;

  if (currentPrice && mostRecentTradingDayClose) {
    const current = parseFloat(currentPrice);
    const reference = parseFloat(mostRecentTradingDayClose);
    if (!isNaN(current) && !isNaN(reference) && reference > 0) {
      mainPriceChange = current - reference;
      mainPriceChangePct = (mainPriceChange / reference) * 100;
      console.log(`Price diff calculation: current=${current}, reference=${reference}, change=${mainPriceChange}, changePct=${mainPriceChangePct.toFixed(2)}%`);
    } else {
      console.warn(`Invalid price values for diff calculation: current=${currentPrice}, reference=${mostRecentTradingDayClose}`);
    }
  } else {
    console.warn(`Cannot calculate price diff: currentPrice=${currentPrice}, mostRecentTradingDayClose=${mostRecentTradingDayClose}`);
    // Fallback to provided change values if available
    if (priceChange !== 0 || priceChangePct !== 0) {
      mainPriceChange = priceChange;
      mainPriceChangePct = priceChangePct;
    }
  }

  if (currentPrice) {
    document.getElementById('mainPrice').textContent = '$' + parseFloat(currentPrice).toFixed(2);
  }

  const changeElement = document.getElementById('mainChange');
  // Always show sign: + for positive, - for negative
  const changeSign = mainPriceChange >= 0 ? '+' : '-';
  const changeColor = mainPriceChange >= 0 ? 'positive' : 'negative';
  changeElement.textContent = `${changeSign}$${Math.abs(mainPriceChange).toFixed(2)} (${changeSign}${Math.abs(mainPriceChangePct).toFixed(2)}%)`;
  changeElement.className = 'change ' + changeColor;

  // -------- Pre-market and After-hours sections --------
  // Compute their deltas on the client to ensure they always use the most recent close.
  // We also decide whether to show the value as "Pre-market" or "After hours"
  // based on the current ET time.

  const preMarketPriceRaw =
    currentPriceData.pre_market_price ||
    currentPriceData.premarket_price ||
    currentPriceData.pre_market;

  const afterHoursPriceRaw =
    currentPriceData.after_hours_price ||
    currentPriceData.extended_hours_price;

  // Reference for pre/post‑market: prefer most_recent_close (yesterday's/today's close),
  // then the main currentPrice, then previousClose.
  const sessionReference =
    (typeof mostRecentClose === 'number' && !isNaN(mostRecentClose) && mostRecentClose > 0
      ? mostRecentClose
      : null) ||
    (currentPrice && !isNaN(parseFloat(currentPrice)) ? parseFloat(currentPrice) : null) ||
    (typeof previousClose === 'number' && !isNaN(previousClose) && previousClose > 0
      ? previousClose
      : null);

  // Determine session type from ET time
  const isPreMarketSession =
    isWeekday && hours >= 4 && ((hours < 9) || (hours === 9 && minutes < 30));
  const isAfterHoursSession =
    isWeekday && hours >= 16 && hours < 20;

  console.log(`[${symbol}] Session reference for pre/after-market:`, {
    preMarketPriceRaw,
    afterHoursPriceRaw,
    sessionReference,
    isPreMarketSession,
    isAfterHoursSession,
    hours,
    minutes
  });

  const preSection = document.getElementById('preMarketSection');
  const afterSection = document.getElementById('afterHoursSection');

  // Decide which raw price to show in which section
  let prePriceForDisplay = null;
  let afterPriceForDisplay = null;

  if (isPreMarketSession) {
    // During pre‑market, show whatever extended‑hours price we have as "Pre‑market"
    prePriceForDisplay = preMarketPriceRaw || afterHoursPriceRaw || null;
  } else if (isAfterHoursSession) {
    // During after‑hours, show as "After hours"
    afterPriceForDisplay = afterHoursPriceRaw || preMarketPriceRaw || null;
  } else {
    // Outside of extended hours, hide both sections
    prePriceForDisplay = null;
    afterPriceForDisplay = null;
  }

  // Helper to compute diff and update a section
  function updateSessionSection(label, priceRaw, refPrice, priceElId, changeElId, sectionEl) {
    if (!priceRaw || !refPrice || !sectionEl) {
      if (sectionEl) sectionEl.style.display = 'none';
      return;
    }
    const price = parseFloat(priceRaw);
    const ref = parseFloat(refPrice);
    if (isNaN(price) || isNaN(ref) || ref <= 0) {
      sectionEl.style.display = 'none';
      return;
    }

    const diff = price - ref;
    const pct = (diff / ref) * 100;
    const sign = diff >= 0 ? '+' : '-';

    const priceEl = document.getElementById(priceElId);
    const changeEl = document.getElementById(changeElId);
    if (priceEl && changeEl) {
      sectionEl.style.display = 'block';
      priceEl.textContent = '$' + price.toFixed(2);
      changeEl.textContent = `${sign}$${Math.abs(diff).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
    }

    console.log(`[${symbol}] ${label} session:`, {
      price,
      ref,
      diff,
      pct
    });
  }

  // Apply updates
  updateSessionSection(
    'Pre-market',
    prePriceForDisplay,
    sessionReference,
    'preMarketPrice',
    'preMarketChange',
    preSection
  );

  updateSessionSection(
    'After-hours',
    afterPriceForDisplay,
    sessionReference,
    'afterHoursPrice',
    'afterHoursChange',
    afterSection
  );

  // Render main metrics
  const mainMetricsGrid = document.getElementById('mainMetricsGrid');
  const mainMetrics = [
    { label: 'Previous Close', value: formatValue(previousClose, true) },
    { label: 'Market Cap (intraday)', value: formatValue(financialInfo.financial_data?.market_cap, true) },
    { label: 'Open', value: formatValue(currentPriceData.open, true) },
    { label: "Day's Range", value: formatValue(currentPriceData.daily_range?.low, true) + ' - ' + formatValue(currentPriceData.daily_range?.high, true) },
    { label: '52 Week Range', value: formatValue(priceInfo.week_52_low, true) + ' - ' + formatValue(priceInfo.week_52_high, true) },
    { label: 'Bid/Ask', value: formatValue(currentPriceData.bid || currentPriceData.bid_price, true) + ' / ' + formatValue(currentPriceData.ask || currentPriceData.ask_price, true) },
    { label: 'Avg. Volume', value: formatValue(financialInfo.financial_data?.average_volume) },
    { label: 'PE Ratio (TTM)', value: formatValue(financialInfo.financial_data?.price_to_earnings || financialInfo.financial_data?.pe_ratio) },
    { label: 'Volume', value: formatValue(currentPriceData.volume || currentPriceData.size) },
    { label: 'EPS (TTM)', value: formatValue(financialInfo.financial_data?.earnings_per_share || financialInfo.financial_data?.eps) },
    { label: 'Earnings Date', value: stockData.earnings_date || 'N/A' }
  ];

  mainMetrics.forEach(metric => {
    if (metric.value && metric.value !== 'N/A' && metric.value !== 'undefined') {
      mainMetricsGrid.appendChild(createMetricCard(metric.label, metric.value));
    }
  });

  // Render financial ratios
  const financialData = financialInfo.financial_data || {};
  if (Object.keys(financialData).length > 0) {
    document.getElementById('financialRatiosSection').style.display = 'block';
    const ratiosGrid = document.getElementById('financialRatiosGrid');

    const ratios = [
      // First row (6 metrics)
      { label: 'EPS (TTM)', value: formatValue(financialData.earnings_per_share || financialData.eps) },
      { label: 'PE Ratio (TTM)', value: formatValue(financialData.price_to_earnings || financialData.pe_ratio) },
      { label: 'Return on Equity (ROE)', value: formatValue(financialData.return_on_equity, false, true) },
      { label: 'Return on Assets (ROA)', value: formatValue(financialData.return_on_assets, false, true) },
      { label: 'Price to Book (P/B)', value: formatValue(financialData.price_to_book) },
      { label: 'Price to Sales (P/S)', value: formatValue(financialData.price_to_sales) },
      // Second row (6 metrics)
      { label: 'Price to Cash Flow (P/CF)', value: formatValue(financialData.price_to_cash_flow) },
      { label: 'Price to Free Cash Flow (P/FCF)', value: formatValue(financialData.price_to_free_cash_flow) },
      { label: 'EV to Sales', value: formatValue(financialData.ev_to_sales) },
      { label: 'EV to EBITDA', value: formatValue(financialData.ev_to_ebitda) },
      { label: 'Enterprise Value', value: formatValue(financialData.enterprise_value, true) },
      { label: 'Free Cash Flow', value: formatValue(financialData.free_cash_flow, true) },
      // Third row (5 metrics)
      { label: 'Current Ratio', value: formatValue(financialData.current || financialData.current_ratio) },
      { label: 'Quick Ratio', value: formatValue(financialData.quick || financialData.quick_ratio) },
      { label: 'Cash Ratio', value: formatValue(financialData.cash || financialData.cash_ratio) },
      { label: 'Debt to Equity', value: formatValue(financialData.debt_to_equity) },
      { label: 'Dividend Yield', value: formatValue(financialData.dividend_yield, false, true) }
    ];

    ratios.forEach(ratio => {
      if (ratio.value && ratio.value !== 'N/A' && ratio.value !== 'undefined') {
        ratiosGrid.appendChild(createMetricCard(ratio.label, ratio.value));
      }
    });
  }

  // Initialize chart and other dynamic features
  // This will be handled by the existing JavaScript code that's already in the template
  // We just need to make the data available globally
  window.stockData = stockData;
  window.allChartData = stockData.chart_data || [];
  window.allChartLabels = stockData.chart_labels || [];
  window.mergedSeries = stockData.merged_series || [];
  window.backendPreviousClose = previousClose;
  window.symbol = symbol;

  // Debug logging
  console.log(`[${symbol}] Chart data setup:`, {
    chart_data_length: window.allChartData.length,
    chart_labels_length: window.allChartLabels.length,
    merged_series_length: window.mergedSeries.length,
    merged_series_sample: window.mergedSeries.slice(0, 3),
    stockData_keys: Object.keys(stockData),
    stockData_merged_series: stockData.merged_series ? stockData.merged_series.length : 'missing'
  });

  // If mergedSeries is empty but stockData has it, try to use it directly
  if (window.mergedSeries.length === 0 && stockData.merged_series && stockData.merged_series.length > 0) {
    console.log(`[${symbol}] mergedSeries was empty, using stockData.merged_series directly`);
    window.mergedSeries = stockData.merged_series;
    window.allChartData = stockData.merged_series.map(s => s.close || s.price || 0);
    window.allChartLabels = stockData.merged_series.map(s => s.timestamp || '');
  }

  // Render options data
  const optionsData = optionsInfo.options_data || {};
  if (optionsData && optionsData.success && optionsData.data && optionsData.data.contracts) {
    const optionsSection = document.getElementById('optionsSection');
    const optionsDisplay = document.getElementById('optionsDisplay');
    if (optionsSection && optionsDisplay) {
      optionsSection.style.display = 'block';

      const contracts = optionsData.data.contracts || [];
      if (contracts.length === 0) {
        optionsDisplay.innerHTML = '<p>No options contracts found</p>';
        return;
      }

      // Group by expiration date
      const byExpiry = {};
      contracts.forEach(contract => {
        const exp = contract.expiration || 'Unknown';
        if (!byExpiry[exp]) {
          byExpiry[exp] = [];
        }
        byExpiry[exp].push(contract);
      });

      const sortedExpirations = Object.keys(byExpiry).sort().slice(0, 10);

      if (sortedExpirations.length === 0) {
        optionsDisplay.innerHTML = '<p>No options contracts found</p>';
        return;
      }

      // Get current price for ATM calculation
      const currentPrice = parseFloat(currentPriceData.price || currentPriceData.close || 0);

      // Build dropdown and tables
      let html = '<div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center;">';
      html += '<div><label for="optionsExpirationSelect" style="margin-right: 8px; font-weight: 600; color: #c9d1d9;">Expiration:</label>';
      html += '<select id="optionsExpirationSelect" onchange="showOptionsForExpiration(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #30363d; border-radius: 4px; background: #0d1117; color: #c9d1d9;">';
      sortedExpirations.forEach((exp, i) => {
        html += `<option value="${exp}" ${i === 0 ? 'selected' : ''}>${exp} (${byExpiry[exp].length} contracts)</option>`;
      });
      html += '</select></div></div>';

      // Build table for each expiration
      sortedExpirations.forEach((expDate, expIdx) => {
        const contractsList = byExpiry[expDate];

        // Group by strike
        const byStrike = {};
        contractsList.forEach(contract => {
          const strike = parseFloat(contract.strike || 0);
          const type = (contract.type || '').toLowerCase();
          if (!byStrike[strike]) {
            byStrike[strike] = { call: null, put: null };
          }
          byStrike[strike][type] = contract;
        });

        const sortedStrikes = Object.keys(byStrike).map(Number).sort((a, b) => b - a);

        const displayStyle = expIdx === 0 ? 'block' : 'none';
        html += `<div id="optionsTable_${expDate}" class="options-table-container" style="display: ${displayStyle}; margin-bottom: 20px;">`;
        html += '<table class="data-table options-chain-table" style="width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; background: #0d1117; border-radius: 8px; overflow: hidden; border: 1px solid #30363d;">';
        html += '<thead><tr>';
        html += '<th colspan="6" style="background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; letter-spacing: 1px; border-right: 2px solid white;">CALLS</th>';
        html += '<th rowspan="2" style="background: linear-gradient(135deg, #5e72e4 0%, #825ee4 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; vertical-align: middle; border-left: 2px solid white; border-right: 2px solid white;">Strike</th>';
        html += '<th colspan="6" style="background: linear-gradient(135deg, #ef5350 0%, #f44336 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; letter-spacing: 1px; border-left: 2px solid white;">PUTS</th>';
        html += '</tr><tr>';
        ['Bid/Ask<br><small style="font-weight: 500; font-size: 11px;">Spread</small>', 'Mid', 'Vol', 'IV', 'Delta<br><small style="font-weight: 500; font-size: 11px;">(Δ)</small>', 'Theta<br><small style="font-weight: 500; font-size: 11px;">(Θ)</small>'].forEach(h => {
          html += `<th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">${h}</th>`;
        });
        ['Bid/Ask<br><small style="font-weight: 500; font-size: 11px;">Spread</small>', 'Mid', 'Vol', 'IV', 'Delta<br><small style="font-weight: 500; font-size: 11px;">(Δ)</small>', 'Theta<br><small style="font-weight: 500; font-size: 11px;">(Θ)</small>'].forEach(h => {
          html += `<th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">${h}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Find ATM strike for coloring
        let atmStrikeIdx = null;
        if (currentPrice && sortedStrikes.length > 0) {
          atmStrikeIdx = sortedStrikes.reduce((minIdx, strike, idx) => {
            return Math.abs(strike - currentPrice) < Math.abs(sortedStrikes[minIdx] - currentPrice) ? idx : minIdx;
          }, 0);
        }

        sortedStrikes.slice(0, 50).forEach((strike, idx) => {
          const call = byStrike[strike].call;
          const put = byStrike[strike].put;

          // Calculate row background color based on moneyness (dark theme)
          const getRowBgColor = (strike, currentPrice, optionType, rowIdx) => {
            const baseColor = rowIdx % 2 === 0 ? '#161b22' : '#0d1117';
            if (!currentPrice) return baseColor;

            const pctDiff = Math.abs(strike - currentPrice) / currentPrice * 100;
            const isCall = optionType === 'call';
            const itm = isCall ? strike < currentPrice : strike > currentPrice;
            const otm = isCall ? strike > currentPrice : strike < currentPrice;

            if (otm) return baseColor;

            // ITM coloring with teal shades (darker for dark theme)
            if (pctDiff < 0.5) return '#004d40';
            if (pctDiff < 1) return '#00695c';
            if (pctDiff < 2) return '#00796b';
            if (pctDiff < 5) return '#00897b';
            if (pctDiff < 10) return '#26a69a';
            return '#4db6ac';
          };

          const callBg = getRowBgColor(strike, currentPrice, 'call', idx);
          const putBg = getRowBgColor(strike, currentPrice, 'put', idx);
          const distanceFromAtm = atmStrikeIdx !== null ? Math.abs(idx - atmStrikeIdx) : 999;

          const cellStyle = (bgColor, align = 'center', isNumeric = false, borderRight = false) => {
            let style = `padding: 12px 10px; background-color: ${bgColor}; text-align: ${align}; font-size: 14px; vertical-align: middle; color: #c9d1d9;`;
            if (borderRight) style += ' border-right: 1px solid #30363d;';
            if (isNumeric) style += ' font-family: "SF Mono", "Monaco", "Courier New", monospace; font-weight: 600;';
            return style;
          };

          html += `<tr class="strike-row" data-distance-from-atm="${distanceFromAtm}" style="transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#e3f2fd'" onmouseout="this.style.backgroundColor=''">`;

          // Call columns
          if (call) {
            const bid = call.bid;
            const ask = call.ask;
            const last = call.last || call.day_close;
            const mid = (bid && ask) ? ((bid + ask) / 2).toFixed(2) : null;
            const spread = (bid && ask) ? (ask - bid).toFixed(2) : null;

            if (bid && ask && bid > 0 && ask > 0) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #66bb6a; font-size: 13px;">$${spread}</strong></td>`;
            } else if (bid) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #8b949e;">-</span></td>`;
            } else if (ask) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #8b949e;">-</span></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #8b949e;">-</span></td>`;
            }

            html += `<td style="${cellStyle(callBg, 'right', true, true)}">${last ? '$' + parseFloat(last).toFixed(2) : '-'}</td>`;
            html += `<td style="${cellStyle(callBg, 'right', true, true)}">${formatValue(call.volume)}</td>`;
            html += `<td style="${cellStyle(callBg, 'right', true, true)}">${formatValue(call.implied_volatility, false, true)}</td>`;
            html += `<td style="${cellStyle(callBg, 'right', true, true)}">${formatValue(call.delta)}</td>`;
            html += `<td style="${cellStyle(callBg, 'right', true, true)}">${formatValue(call.theta)}</td>`;
          } else {
            html += `<td colspan="6" style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
          }

          // Strike column
          html += `<td style="padding: 12px 10px; background-color: #667eea; color: white; text-align: center; font-weight: 700; font-size: 14px; border-left: 2px solid white; border-right: 2px solid white;">${formatValue(strike, true)}</td>`;

          // Put columns
          if (put) {
            const bid = put.bid;
            const ask = put.ask;
            const last = put.last || put.day_close;
            const mid = (bid && ask) ? ((bid + ask) / 2).toFixed(2) : null;
            const spread = (bid && ask) ? (ask - bid).toFixed(2) : null;

            if (bid && ask && bid > 0 && ask > 0) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #ef5350; font-size: 13px;">$${spread}</strong></td>`;
            } else if (bid) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #8b949e;">-</span></td>`;
            } else if (ask) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #c9d1d9; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #8b949e;">-</span></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #8b949e;">-</span></td>`;
            }

            html += `<td style="${cellStyle(putBg, 'right', true, true)}">${last ? '$' + parseFloat(last).toFixed(2) : '-'}</td>`;
            html += `<td style="${cellStyle(putBg, 'right', true, true)}">${formatValue(put.volume)}</td>`;
            html += `<td style="${cellStyle(putBg, 'right', true, true)}">${formatValue(put.implied_volatility, false, true)}</td>`;
            html += `<td style="${cellStyle(putBg, 'right', true, true)}">${formatValue(put.delta)}</td>`;
            html += `<td style="${cellStyle(putBg, 'right', true, true)}">${formatValue(put.theta)}</td>`;
          } else {
            html += `<td colspan="6" style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
          }

          html += '</tr>';
        });

        html += '</tbody></table></div>';
      });

      optionsDisplay.innerHTML = html;
      console.log(`Options rendered: ${sortedExpirations.length} expirations, ${contracts.length} total contracts`);
    }
  } else if (optionsInfo && optionsInfo.error) {
    const optionsSection = document.getElementById('optionsSection');
    const optionsDisplay = document.getElementById('optionsDisplay');
    if (optionsSection && optionsDisplay) {
      optionsSection.style.display = 'block';
      optionsDisplay.innerHTML = `<p style="color: #f85149;">Error loading options: ${optionsInfo.error}</p>`;
    }
  }

  // Render news data
  const newsData = newsInfo.news_data || {};
  const newsArticles = newsData.articles || [];
  if (newsArticles && newsArticles.length > 0) {
    const newsSection = document.getElementById('newsSection');
    const newsDisplay = document.getElementById('newsDisplay');
    if (newsSection && newsDisplay) {
      newsSection.style.display = 'block';
      let newsHTML = '<ul style="list-style: none; padding: 0;">';
      newsArticles.slice(0, 10).forEach(article => {
        const title = article.title || 'No title';
        const published = article.published_utc ? article.published_utc.substring(0, 10) : '';
        const description = article.description || '';
        const url = article.article_url || '#';
        const descSnippet = description.length > 200 ? description.substring(0, 200) + '...' : description;
        newsHTML += `<li style="margin-bottom: 15px; padding: 10px; background: #161b22; border-radius: 4px; border: 1px solid #30363d;">
          <strong style="color: #f0f6fc;">${title}</strong><br>
          <small style="color: #8b949e;">${published}</small><br>
          ${descSnippet ? `<p style="color: #c9d1d9; margin: 5px 0;">${descSnippet}</p>` : ''}
          <a href="${url}" target="_blank" style="color: #667eea;">Read more</a>
        </li>`;
      });
      newsHTML += '</ul>';
      newsDisplay.innerHTML = newsHTML;
    }
  }

  // Render IV data
  const ivData = ivInfo.iv_data || {};
  if (ivData && Object.keys(ivData).length > 0) {
    const ivSection = document.getElementById('ivSection');
    const ivTable = document.getElementById('ivTable');
    if (ivSection && ivTable) {
      ivSection.style.display = 'block';
      // Add IV metrics to table
      const ivMetrics = [
        { label: 'Mean IV', value: ivData.statistics?.mean || ivData.atm_iv?.mean },
        { label: 'Median IV', value: ivData.statistics?.median || ivData.atm_iv?.median },
        { label: 'Min IV', value: ivData.statistics?.min || ivData.atm_iv?.min },
        { label: 'Max IV', value: ivData.statistics?.max || ivData.atm_iv?.max },
        { label: 'Std Dev', value: ivData.statistics?.std_dev || ivData.atm_iv?.std_dev }
      ];
      ivMetrics.forEach(metric => {
        if (metric.value !== undefined && metric.value !== null) {
          const row = ivTable.insertRow();
          row.insertCell(0).textContent = metric.label;
          row.insertCell(1).textContent = typeof metric.value === 'number' ? metric.value.toFixed(2) : metric.value;
        }
      });
    }
  }

  // Trigger chart initialization if the chart script is loaded
  if (typeof Chart !== 'undefined' && document.getElementById('priceChart')) {
    // Chart initialization will be handled by existing code
    console.log('Chart data loaded, ready for initialization');
  }

  console.log('Stock data rendered successfully');
})();

