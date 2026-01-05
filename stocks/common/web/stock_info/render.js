// Stock Info Renderer
// This script renders dynamic data into the static template

(function () {
  'use strict';

  // Check for debug flag in URL query parameters
  const urlParams = new URLSearchParams(window.location.search);
  const debugMode = urlParams.get('debug') === 'true';
  
  // Debug logging function - only logs when debug=true in URL
  function debugLog(...args) {
    if (debugMode) {
      console.log(...args);
    }
  }

  // Get JSON data from embedded script tag
  const dataScript = document.getElementById('stockData');
  if (!dataScript || !dataScript.textContent) {
    console.error('No stock data found - stockData element missing or empty');
    console.error('Available script elements:', Array.from(document.querySelectorAll('script')).map(s => s.id || 'no-id'));
    return;
  }

  let stockData;
  try {
    const jsonText = dataScript.textContent.trim();
    debugLog('Parsing stock data, length:', jsonText.length, 'first 200 chars:', jsonText.substring(0, 200));
    stockData = JSON.parse(jsonText);
    debugLog('Stock data parsed successfully, keys:', Object.keys(stockData));
  } catch (e) {
    console.error('Failed to parse stock data:', e);
    console.error('Data content (first 500 chars):', dataScript.textContent.substring(0, 500));
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
    debugLog(`[${symbol}] Market CLOSED - currentPrice determined:`, {
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

  debugLog(`[${symbol}] Price diff calculation setup:`, {
    currentPrice: currentPrice,
    previousClose: previousClose,
    most_recent_trading_day_close: currentPriceData.most_recent_trading_day_close,
    day_before_last_trading_day_close: currentPriceData.day_before_last_trading_day_close,
    mostRecentTradingDayClose: mostRecentTradingDayClose,
    isMarketClosed: isMarketClosed
  });

  const priceChange = currentPriceData.change || currentPriceData.change_amount || 0;
  const priceChangePct = currentPriceData.change_percent || currentPriceData.change_pct || 0;

  debugLog(`[${symbol}] Raw price fields:`, {
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
      debugLog(`Price diff calculation: current=${current}, reference=${reference}, change=${mainPriceChange}, changePct=${mainPriceChangePct.toFixed(2)}%`);
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
    currentPriceData.after_hours ||
    currentPriceData.extended_hours_price ||
    currentPriceData.extended_hours;

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
  // After-hours: after 4 PM on weekdays, or anytime on weekends (when market is closed)
  // But not during pre-market hours
  // Also include late night (after 8 PM) as after-hours
  const isAfterHoursSession =
    (isMarketClosed && !isPreMarketSession) || (isWeekday && hours >= 20);

  debugLog(`[${symbol}] Session reference for pre/after-market:`, {
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
    // Always show during pre-market, even if no data (will show "nan")
    // Check if we have any price data, otherwise use NaN to trigger "nan" display
    if (preMarketPriceRaw || afterHoursPriceRaw) {
      prePriceForDisplay = preMarketPriceRaw || afterHoursPriceRaw;
    } else {
      // No data available, but we're in pre-market, so show "nan"
      prePriceForDisplay = NaN;
    }
    // Never show after-hours during pre-market
    afterPriceForDisplay = null;
    
    debugLog(`[${symbol}] Pre-market session detected:`, {
      isPreMarketSession,
      hours,
      minutes,
      preMarketPriceRaw,
      afterHoursPriceRaw,
      prePriceForDisplay
    });
  } else if (isAfterHoursSession) {
    // During after‑hours, show as "After hours"
    // Always show during after-hours, even if no data (will show "nan")
    if (afterHoursPriceRaw || preMarketPriceRaw) {
      afterPriceForDisplay = afterHoursPriceRaw || preMarketPriceRaw;
  } else {
      // No data available, but we're in after-hours, so show "nan"
      afterPriceForDisplay = NaN;
    }
    // Never show pre-market during after-hours
    prePriceForDisplay = null;
    
    debugLog(`[${symbol}] After-hours session detected:`, {
      isAfterHoursSession,
      hours,
      minutes,
      isMarketClosed,
      afterHoursPriceRaw,
      preMarketPriceRaw,
      afterPriceForDisplay
    });
  } else {
    // Market is open (regular trading hours) - don't show extended hours sections
    // Never show pre-market or after-hours during regular market hours
    prePriceForDisplay = null;
    afterPriceForDisplay = null;
  }

  // Helper to compute diff and update a section
  function updateSessionSection(label, priceRaw, refPrice, priceElId, changeElId, sectionEl, timeElId) {
    if (!sectionEl) {
      return;
    }
    
    const priceEl = document.getElementById(priceElId);
    const changeEl = document.getElementById(changeElId);
    const timeEl = timeElId ? document.getElementById(timeElId) : null;
    
    if (!priceEl || !changeEl) {
      sectionEl.style.display = 'none';
      return;
    }
    
    // If priceRaw is null/undefined (not explicitly NaN), hide the section
    if (priceRaw === null || priceRaw === undefined) {
      sectionEl.style.display = 'none';
      return;
    }
    
    // Show section even if price is NaN (like dynamic version shows "nan $nan (nan%)")
    // This allows showing "nan" when explicitly set to NaN (e.g., during after-hours with no data)
    const price = parseFloat(priceRaw);
    const ref = parseFloat(refPrice);
    
    if (isNaN(price)) {
      // Show "nan" like the dynamic version when price is explicitly NaN
      sectionEl.style.display = 'block';
      priceEl.textContent = 'nan';
      changeEl.textContent = '$nan (nan%)';
      if (timeEl) {
        const now = new Date();
        const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
        timeEl.textContent = et.toLocaleTimeString('en-US', {
          hour: 'numeric',
          minute: '2-digit',
          second: '2-digit',
          timeZone: 'America/New_York',
          hour12: true
        }) + ' EST';
      }
      debugLog(`[${symbol}] ${label} session: price is NaN, showing "nan"`);
      return;
    }
    
    if (isNaN(ref) || ref <= 0) {
      // If we have a price but no valid reference, just show the price
      sectionEl.style.display = 'block';
      priceEl.textContent = '$' + price.toFixed(2);
      changeEl.textContent = '--';
      if (timeEl) {
        const now = new Date();
        const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
        timeEl.textContent = et.toLocaleTimeString('en-US', {
          hour: 'numeric',
          minute: '2-digit',
          second: '2-digit',
          timeZone: 'America/New_York',
          hour12: true
        }) + ' EST';
      }
      debugLog(`[${symbol}] ${label} session: no valid reference price`);
      return;
    }

    const diff = price - ref;
    const pct = (diff / ref) * 100;
    const sign = diff >= 0 ? '+' : '-';

      sectionEl.style.display = 'block';
      priceEl.textContent = '$' + price.toFixed(2);
      changeEl.textContent = `${sign}$${Math.abs(diff).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
    
    // Apply color classes based on change direction (green for positive, red for negative)
    changeEl.classList.remove('positive', 'negative');
    if (diff > 0) {
      changeEl.classList.add('positive');
    } else if (diff < 0) {
      changeEl.classList.add('negative');
    }
    // If diff === 0, no class is added (neutral/default color)
    
    if (timeEl) {
      const now = new Date();
      const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
      timeEl.textContent = et.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
        timeZone: 'America/New_York',
        hour12: true
      }) + ' EST';
    }

    debugLog(`[${symbol}] ${label} session:`, {
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
    preSection,
    'preMarketTime'
  );

  updateSessionSection(
    'After-hours',
    afterPriceForDisplay,
    sessionReference,
    'afterHoursPrice',
    'afterHoursChange',
    afterSection,
    'afterHoursTime'
  );

  // Calculate 52-week range from mergedSeries if not available in priceInfo
  let week52Low = priceInfo.week_52_low;
  let week52High = priceInfo.week_52_high;
  
  // Check if values are actually valid numbers (not null, undefined, 0, or NaN)
  const hasValidWeek52Low = week52Low !== null && week52Low !== undefined && !isNaN(week52Low) && week52Low > 0;
  const hasValidWeek52High = week52High !== null && week52High !== undefined && !isNaN(week52High) && week52High > 0;
  
  debugLog(`[${symbol}] 52-week range check:`, {
    week52Low,
    week52High,
    hasValidWeek52Low,
    hasValidWeek52High,
    mergedSeriesLength: window.mergedSeries ? window.mergedSeries.length : 0
  });
  
  if ((!hasValidWeek52Low || !hasValidWeek52High) && window.mergedSeries && Array.isArray(window.mergedSeries) && window.mergedSeries.length > 0) {
    // Calculate from mergedSeries - get last 365 days
    const oneYearAgo = Date.now() - (365 * 24 * 60 * 60 * 1000);
    const prices = [];
    
    for (const record of window.mergedSeries) {
      if (!record || typeof record !== 'object') continue;
      const timestamp = record.timestamp;
      const close = record.close || record.price;
      
      if (!timestamp || !close) continue;
      
      // Parse timestamp
      const recordDate = new Date(timestamp);
      if (isNaN(recordDate.getTime())) continue;
      
      // Only include records from last 365 days
      if (recordDate.getTime() >= oneYearAgo) {
        const price = parseFloat(close);
        if (!isNaN(price) && price > 0) {
          prices.push(price);
        }
      }
    }
    
    debugLog(`[${symbol}] Calculated 52-week range from mergedSeries:`, {
      pricesFound: prices.length,
      minPrice: prices.length > 0 ? Math.min(...prices) : null,
      maxPrice: prices.length > 0 ? Math.max(...prices) : null
    });
    
    if (prices.length > 0) {
      if (!hasValidWeek52Low) week52Low = Math.min(...prices);
      if (!hasValidWeek52High) week52High = Math.max(...prices);
    }
  }
  
  // Format Day's Range - always show, even if N/A
  const dailyRangeLow = currentPriceData.daily_range?.low;
  const dailyRangeHigh = currentPriceData.daily_range?.high;
  let daysRangeValue = 'N/A - N/A';
  if (dailyRangeLow !== null && dailyRangeLow !== undefined && dailyRangeHigh !== null && dailyRangeHigh !== undefined) {
    daysRangeValue = formatValue(dailyRangeLow, true) + ' - ' + formatValue(dailyRangeHigh, true);
  } else if (dailyRangeLow !== null && dailyRangeLow !== undefined) {
    daysRangeValue = formatValue(dailyRangeLow, true) + ' - N/A';
  } else if (dailyRangeHigh !== null && dailyRangeHigh !== undefined) {
    daysRangeValue = 'N/A - ' + formatValue(dailyRangeHigh, true);
  }
  
  // Format 52 Week Range - always show, even if N/A
  let week52RangeValue = 'N/A - N/A';
  // Check if values are valid (not null, undefined, NaN, or 0)
  const hasValidLow = week52Low !== null && week52Low !== undefined && !isNaN(week52Low) && isFinite(week52Low) && week52Low > 0;
  const hasValidHigh = week52High !== null && week52High !== undefined && !isNaN(week52High) && isFinite(week52High) && week52High > 0;
  
  debugLog(`[${symbol}] 52-week range formatting:`, {
    week52Low,
    week52High,
    hasValidLow,
    hasValidHigh,
    priceInfoKeys: Object.keys(priceInfo),
    priceInfoWeek52: { low: priceInfo.week_52_low, high: priceInfo.week_52_high }
  });
  
  if (hasValidLow && hasValidHigh) {
    week52RangeValue = formatValue(week52Low, true) + ' - ' + formatValue(week52High, true);
    debugLog(`[${symbol}] 52-week range formatted: ${week52RangeValue}`);
  } else if (hasValidLow) {
    week52RangeValue = formatValue(week52Low, true) + ' - N/A';
  } else if (hasValidHigh) {
    week52RangeValue = 'N/A - ' + formatValue(week52High, true);
  } else {
    console.warn(`[${symbol}] 52-week range not available - week52Low=${week52Low}, week52High=${week52High}`);
  }

  // Render main metrics
  const mainMetricsGrid = document.getElementById('mainMetricsGrid');
  const mainMetrics = [
    { label: 'Previous Close', value: formatValue(previousClose, true) },
    { label: 'Market Cap (intraday)', value: formatValue(financialInfo.financial_data?.market_cap, true) },
    { label: 'Open', value: formatValue(currentPriceData.open, true) },
    { label: "Day's Range", value: daysRangeValue, alwaysShow: true },
    { label: '52 Week Range', value: week52RangeValue, alwaysShow: true },
    { label: 'Bid/Ask', value: formatValue(currentPriceData.bid || currentPriceData.bid_price, true) + ' / ' + formatValue(currentPriceData.ask || currentPriceData.ask_price, true) },
    { label: 'Avg. Volume', value: formatValue(financialInfo.financial_data?.average_volume) },
    { label: 'PE Ratio (TTM)', value: formatValue(financialInfo.financial_data?.price_to_earnings || financialInfo.financial_data?.pe_ratio) },
    { label: 'Volume', value: formatValue(currentPriceData.volume || currentPriceData.size) },
    { label: 'EPS (TTM)', value: formatValue(financialInfo.financial_data?.earnings_per_share || financialInfo.financial_data?.eps) },
    { label: 'Earnings Date', value: stockData.earnings_date || 'N/A' }
  ];

  mainMetrics.forEach(metric => {
    // Always show metrics marked with alwaysShow, or if they have a valid value
    if (metric.alwaysShow || (metric.value && metric.value !== 'N/A' && metric.value !== 'undefined' && !metric.value.includes('undefined'))) {
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
    
    // IV Analysis Section - match dynamic version
    const hasIVData = financialData.iv_30d !== null && financialData.iv_30d !== undefined ||
                     financialData.iv_rank !== null && financialData.iv_rank !== undefined ||
                     financialData.iv_90d_rank !== null && financialData.iv_90d_rank !== undefined ||
                     financialData.relative_rank !== null && financialData.relative_rank !== undefined ||
                     financialData.iv_strategy?.recommendation ||
                     financialData.iv_strategy?.risk_score !== null && financialData.iv_strategy?.risk_score !== undefined ||
                     financialData.iv_metrics?.hv_1yr_range ||
                     financialData.iv_metrics?.roll_yield;
    
    if (hasIVData) {
      // Add separator and header
      const separator = document.createElement('div');
      separator.style.cssText = 'grid-column: 1 / -1; margin: 20px 0 10px 0; border-top: 2px solid #30363d; padding-top: 15px;';
      const header = document.createElement('h3');
      header.textContent = 'IV ANALYSIS';
      header.style.cssText = 'margin: 0; font-size: 16px; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px;';
      separator.appendChild(header);
      ratiosGrid.appendChild(separator);
      
      // IV Analysis bar (gradient card) - shows 30-day IV
      if (financialData.iv_30d !== null && financialData.iv_30d !== undefined) {
        const ivBarCard = document.createElement('div');
        ivBarCard.className = 'metric-card';
        ivBarCard.style.cssText = 'grid-column: span 2; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: 2px solid #5a67d8;';
        const ivLabel = document.createElement('div');
        ivLabel.className = 'metric-label';
        ivLabel.textContent = 'IV Analysis';
        ivLabel.style.cssText = 'color: white; font-weight: bold; font-size: 14px;';
        const ivValue = document.createElement('div');
        ivValue.className = 'metric-value';
        ivValue.textContent = formatValue(financialData.iv_30d, false, true);
        ivValue.style.cssText = 'color: white; font-size: 12px; margin-top: 5px;';
        ivBarCard.appendChild(ivLabel);
        ivBarCard.appendChild(ivValue);
        // Add 30-day label to clarify what the main value represents
        const iv30dLabel = document.createElement('div');
        iv30dLabel.textContent = `30-day: ${formatValue(financialData.iv_30d, false, true)}`;
        iv30dLabel.style.cssText = 'color: white; font-size: 11px; margin-top: 3px; opacity: 0.9;';
        ivBarCard.appendChild(iv30dLabel);
        // Add 90-day if available
        // Add 90-day if available
        if (financialData.iv_90d !== null && financialData.iv_90d !== undefined) {
          const iv90d = document.createElement('div');
          iv90d.textContent = `90-day: ${formatValue(financialData.iv_90d, false, true)}`;
          iv90d.style.cssText = 'color: white; font-size: 11px; margin-top: 3px; opacity: 0.9;';
          ivBarCard.appendChild(iv90d);
        }
        ratiosGrid.appendChild(ivBarCard);
      }
      
      // IV Rank (30-day)
      if (financialData.iv_rank !== null && financialData.iv_rank !== undefined) {
        ratiosGrid.appendChild(createMetricCard('IV Rank (30-day)', formatValue(financialData.iv_rank)));
      }
      
      // IV Rank (90-day)
      const iv90dRank = financialData.iv_90d_rank || financialData.iv_metrics?.rank_90d;
      if (iv90dRank !== null && iv90dRank !== undefined) {
        ratiosGrid.appendChild(createMetricCard('IV Rank (90-day)', formatValue(iv90dRank)));
      }
      
      // Rank Ratio (30d/90d)
      const rankDiff = financialData.iv_rank_diff || financialData.iv_metrics?.rank_diff;
      if (rankDiff !== null && rankDiff !== undefined) {
        const ratioCard = createMetricCard('Rank Ratio (30d/90d)', formatValue(rankDiff));
        const valueEl = ratioCard.querySelector('.metric-value');
        if (valueEl) {
          const ratioValue = parseFloat(rankDiff);
          valueEl.style.color = ratioValue > 1.0 ? '#ef4444' : ratioValue < 1.0 ? '#10b981' : '#6b7280';
        }
        ratiosGrid.appendChild(ratioCard);
      }
      
      // Relative Rank (vs VOO)
      if (financialData.relative_rank !== null && financialData.relative_rank !== undefined) {
        ratiosGrid.appendChild(createMetricCard('Relative Rank (vs VOO)', formatValue(financialData.relative_rank)));
      }
      
      // IV Recommendation
      if (financialData.iv_strategy?.recommendation) {
        const recCard = document.createElement('div');
        recCard.className = 'metric-card';
        recCard.style.cssText = 'grid-column: span 2;';
        const recLabel = document.createElement('div');
        recLabel.className = 'metric-label';
        recLabel.textContent = 'IV Recommendation';
        const recValue = document.createElement('div');
        recValue.className = 'metric-value';
        recValue.textContent = financialData.iv_strategy.recommendation;
        const rec = financialData.iv_strategy.recommendation;
        recValue.style.cssText = `font-weight: bold; color: ${rec === 'BUY LEAP' ? '#10b981' : rec.includes('SELL') ? '#ef4444' : '#6b7280'};`;
        recCard.appendChild(recLabel);
        recCard.appendChild(recValue);
        if (financialData.iv_strategy.notes?.meaning) {
          const meaning = document.createElement('div');
          meaning.textContent = financialData.iv_strategy.notes.meaning;
          meaning.style.cssText = 'font-size: 11px; color: #6b7280; margin-top: 5px;';
          recCard.appendChild(meaning);
        }
        ratiosGrid.appendChild(recCard);
      }
      
      // Risk Score
      if (financialData.iv_strategy?.risk_score !== null && financialData.iv_strategy?.risk_score !== undefined) {
        ratiosGrid.appendChild(createMetricCard('Risk Score', formatValue(financialData.iv_strategy.risk_score)));
      }
      
      // HV 1Y Range
      if (financialData.iv_metrics?.hv_1yr_range) {
        const hvCard = createMetricCard('HV 1Y Range', financialData.iv_metrics.hv_1yr_range);
        const hvValue = hvCard.querySelector('.metric-value');
        if (hvValue) {
          hvValue.style.fontSize = '11px';
        }
        ratiosGrid.appendChild(hvCard);
      }
      
      // Roll Yield
      if (financialData.iv_metrics?.roll_yield !== null && financialData.iv_metrics?.roll_yield !== undefined) {
        ratiosGrid.appendChild(createMetricCard('Roll Yield', formatValue(financialData.iv_metrics.roll_yield, false, true)));
      }
    }
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
  debugLog(`[${symbol}] Chart data setup:`, {
    chart_data_length: window.allChartData.length,
    chart_labels_length: window.allChartLabels.length,
    merged_series_length: window.mergedSeries.length,
    merged_series_sample: window.mergedSeries.slice(0, 3),
    stockData_keys: Object.keys(stockData),
    stockData_merged_series: stockData.merged_series ? stockData.merged_series.length : 'missing'
  });

  // If mergedSeries is empty but stockData has it, try to use it directly
  if (window.mergedSeries.length === 0 && stockData.merged_series && stockData.merged_series.length > 0) {
    debugLog(`[${symbol}] mergedSeries was empty, using stockData.merged_series directly`);
    window.mergedSeries = stockData.merged_series;
    window.allChartData = stockData.merged_series.map(s => s.close || s.price || 0);
    window.allChartLabels = stockData.merged_series.map(s => s.timestamp || '');
  }
  
  // Trigger chart initialization if it's waiting for data
  // This ensures the chart initializes after render.js sets the data
  if (window.mergedSeries && window.mergedSeries.length > 0) {
    debugLog(`[${symbol}] Data is ready, triggering chart initialization`);
    // Give a small delay to ensure all scripts are loaded
    setTimeout(() => {
      if (typeof window.tryInitChart === 'function') {
        debugLog(`[${symbol}] Calling window.tryInitChart()`);
        window.tryInitChart();
      } else if (typeof window.initChart === 'function') {
        debugLog(`[${symbol}] Calling window.initChart()`);
        window.initChart();
      } else {
        console.warn(`[${symbol}] Chart initialization functions not found - tryInitChart: ${typeof window.tryInitChart}, initChart: ${typeof window.initChart}`);
      }
    }, 200);
  } else {
    console.warn(`[${symbol}] No mergedSeries data available for chart:`, {
      hasMergedSeries: !!window.mergedSeries,
      length: window.mergedSeries ? window.mergedSeries.length : 0,
      stockDataKeys: Object.keys(stockData),
      hasMergedSeriesInStockData: !!stockData.merged_series,
      mergedSeriesLength: stockData.merged_series ? stockData.merged_series.length : 0
    });
  }
  
  // Trigger chart initialization if it's waiting for data
  // This ensures the chart initializes after render.js sets the data
  if (window.mergedSeries && window.mergedSeries.length > 0) {
    debugLog(`[${symbol}] Data is ready, triggering chart initialization`);
    // Give a small delay to ensure all scripts are loaded
    setTimeout(() => {
      if (typeof window.tryInitChart === 'function') {
        window.tryInitChart();
      } else if (typeof window.initChart === 'function') {
        window.initChart();
      } else {
        console.warn('Chart initialization functions not found');
      }
    }, 200);
  } else {
    console.warn(`[${symbol}] No mergedSeries data available for chart:`, {
      hasMergedSeries: !!window.mergedSeries,
      length: window.mergedSeries ? window.mergedSeries.length : 0,
      stockDataKeys: Object.keys(stockData),
      hasMergedSeriesInStockData: !!stockData.merged_series
    });
  }

  // Render options data
  const optionsData = optionsInfo.options_data || {};
  debugLog(`[${symbol}] Options data structure:`, {
    hasOptionsData: !!optionsData,
    hasSuccess: optionsData.success !== undefined,
    success: optionsData.success,
    hasData: !!optionsData.data,
    hasContracts: !!(optionsData.data && optionsData.data.contracts),
    contractsCount: optionsData.data?.contracts?.length || 0,
    optionsDataKeys: Object.keys(optionsData)
  });
  
  // Handle different options data structures
  let contracts = [];
  if (optionsData) {
    if (optionsData.success && optionsData.data && optionsData.data.contracts) {
      // Standard structure: {success: true, data: {contracts: [...]}}
      contracts = optionsData.data.contracts;
    } else if (Array.isArray(optionsData.contracts)) {
      // Alternative structure: {contracts: [...]}
      contracts = optionsData.contracts;
    } else if (Array.isArray(optionsData)) {
      // Direct array of contracts
      contracts = optionsData;
    }
  }
  
  if (contracts.length > 0) {
    const optionsSection = document.getElementById('optionsSection');
    const optionsDisplay = document.getElementById('optionsDisplay');
    if (optionsSection && optionsDisplay) {
      optionsSection.style.display = 'block';

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

      // Get current price for ATM calculation - try multiple sources
      const mainPriceEl = document.getElementById('mainPrice');
      const currentPriceFromPage = mainPriceEl ? parseFloat(mainPriceEl.textContent.replace('$', '').replace(',', '')) : null;
      const currentPrice = currentPriceFromPage || 
                          parseFloat(currentPriceData.price || currentPriceData.close || currentPriceData.last_price || 0);
      
      debugLog(`[${symbol}] Options current price:`, {
        fromPage: currentPriceFromPage,
        fromData: currentPriceData.price || currentPriceData.close,
        final: currentPrice
      });

      // Build dropdown and tables
      let html = '<div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center;">';
      html += '<div><label for="optionsExpirationSelect" style="margin-right: 8px; font-weight: 600;">Expiration:</label>';
      html += '<select id="optionsExpirationSelect" onchange="showOptionsForExpiration(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">';
      sortedExpirations.forEach((exp, i) => {
        html += `<option value="${exp}" ${i === 0 ? 'selected' : ''}>${exp} (${byExpiry[exp].length} contracts)</option>`;
      });
      html += '</select></div>';
      
      // Add strike range dropdown
      html += '<div><label for="strikeRangeSelect" style="margin-right: 8px; font-weight: 600;">Show strikes:</label>';
      html += '<select id="strikeRangeSelect" onchange="filterStrikesByRange(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">';
      html += '<option value="10" selected>±10 around ATM</option>';
      html += '<option value="15">±15 around ATM</option>';
      html += '<option value="20">±20 around ATM</option>';
      html += '<option value="all">All strikes</option>';
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

        // Sort strikes descending (high to low) for display
        const sortedStrikes = Object.keys(byStrike).map(Number).sort((a, b) => b - a);
        
        // Debug: log strikes and current price
        debugLog(`[${symbol}] Options filtering:`, {
          currentPrice,
          sortedStrikes: sortedStrikes.slice(0, 10),
          totalStrikes: sortedStrikes.length,
          atmStrikeIdx: null // Will be set below
        });

        const displayStyle = expIdx === 0 ? 'block' : 'none';
        html += `<div id="optionsTable_${expDate}" class="options-table-container" style="display: ${displayStyle}; margin-bottom: 20px;">`;
        html += '<table class="data-table options-chain-table" style="width: 100%; margin-bottom: 20px; font-size: 14px; border-collapse: separate; border-spacing: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; background: #ffffff;">';
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

        // Find ATM strike (closest strike to current price) for coloring and filtering
        let atmStrike = null;
        let atmStrikeIdx = null;
        if (currentPrice && sortedStrikes.length > 0) {
          // Find the strike closest to current price
          atmStrikeIdx = sortedStrikes.reduce((minIdx, strike, idx) => {
            return Math.abs(strike - currentPrice) < Math.abs(sortedStrikes[minIdx] - currentPrice) ? idx : minIdx;
          }, 0);
          atmStrike = sortedStrikes[atmStrikeIdx];
        }

        // Filter strikes to show around ATM, then apply ±N filter
        // First, find strikes around ATM (within reasonable range, e.g., ±$50)
        let strikesToShow = sortedStrikes;
        if (currentPrice && atmStrike !== null) {
          // Show strikes within ±$50 of current price, or all if less than 100 strikes total
          const priceRange = 50;
          strikesToShow = sortedStrikes.filter(strike => Math.abs(strike - currentPrice) <= priceRange);
          // If we filtered too much, show at least 50 strikes around ATM
          if (strikesToShow.length < 50 && sortedStrikes.length > 50) {
            // Get index range around ATM
            const startIdx = Math.max(0, atmStrikeIdx - 25);
            const endIdx = Math.min(sortedStrikes.length, atmStrikeIdx + 25);
            strikesToShow = sortedStrikes.slice(startIdx, endIdx);
          }
          // If still too few, just show all (up to 100)
          if (strikesToShow.length === 0) {
            strikesToShow = sortedStrikes.slice(0, 100);
          }
        } else {
          // No current price, just show first 100
          strikesToShow = sortedStrikes.slice(0, 100);
        }
        
        strikesToShow.forEach((strike, displayIdx) => {
          // Find the original index in sortedStrikes for distance calculation
          const originalIdx = sortedStrikes.indexOf(strike);
          const idx = originalIdx >= 0 ? originalIdx : displayIdx;
          const call = byStrike[strike].call;
          const put = byStrike[strike].put;

          // Calculate row background color based on moneyness and distance from ATM
          const getRowBgColor = (strike, currentPrice, optionType, rowIdx, distanceFromAtm) => {
            const baseColor = rowIdx % 2 === 0 ? '#ffffff' : '#f5f5f5';
            if (!currentPrice) return baseColor;

            const pctDiff = Math.abs(strike - currentPrice) / currentPrice * 100;
            const isCall = optionType === 'call';
            const itm = isCall ? strike < currentPrice : strike > currentPrice;

            // Highlight ATM rows (within 2% of current price) with light blue tint
            if (pctDiff < 2) {
              return '#e3f2fd'; // Light blue for ATM
            }

            if (!itm) return baseColor;

            // ITM coloring - very light yellow tint
            return '#fffbf0';
          };

          // Calculate distance from ATM in terms of number of strikes away (not price difference)
          // This ensures ±10 means 10 strikes above and 10 strikes below the ATM strike
          const distanceFromAtm = atmStrikeIdx !== null ? Math.abs(idx - atmStrikeIdx) : 999;
          const callBg = getRowBgColor(strike, currentPrice, 'call', idx, distanceFromAtm);
          const putBg = getRowBgColor(strike, currentPrice, 'put', idx, distanceFromAtm);

          const cellStyle = (bgColor, align = 'center', isNumeric = false, borderRight = false) => {
            let style = `padding: 12px 10px; background-color: ${bgColor}; text-align: ${align}; font-size: 14px; vertical-align: middle; color: #1a1a1a;`;
            if (borderRight) style += ' border-right: 1px solid #d0d0d0;';
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
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #1b5e20; font-size: 13px;">$${spread}</strong></td>`;
            } else if (bid) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #666;">-</span></td>`;
            } else if (ask) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #666;">-</span></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }

            // Mid column
            if (mid) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><strong style="color: #0d47a1; font-size: 15px;">$${mid}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Vol column
            const vol = call.volume;
            if (typeof vol === 'number' && vol > 0) {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}">${vol.toLocaleString()}</td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // IV column
            const iv = call.implied_volatility;
            if (typeof iv === 'number') {
              html += `<td style="${cellStyle(callBg, 'right', true, true)}"><strong style="color: #1a1a1a;">${(iv * 100).toFixed(1)}%</strong></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Delta column
            const delta = call.delta;
            if (typeof delta === 'number') {
              html += `<td style="${cellStyle(callBg, 'center', true, true)}"><strong style="color: #004d40; font-size: 15px;">${delta.toFixed(3)}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Theta column
            const theta = call.theta;
            if (typeof theta === 'number') {
              html += `<td style="${cellStyle(callBg, 'center', true, false)}"><strong style="color: #bf360c; font-size: 15px;">${theta.toFixed(3)}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(callBg, 'center', false, false)}"><span style="color: #666;">-</span></td>`;
            }
          } else {
            html += `<td colspan="6" style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
          }

          // Strike column
          html += `<td style="padding: 12px 10px; background-color: #5e72e4; color: white; text-align: center; font-weight: 700; font-size: 14px; border-left: 2px solid white; border-right: 2px solid white;">$${strike.toFixed(2)}</td>`;

          // Put columns
          if (put) {
            const bid = put.bid;
            const ask = put.ask;
            const last = put.last || put.day_close;
            const mid = (bid && ask) ? ((bid + ask) / 2).toFixed(2) : null;
            const spread = (bid && ask) ? (ask - bid).toFixed(2) : null;

            if (bid && ask && bid > 0 && ask > 0) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #b71c1c; font-size: 13px;">$${spread}</strong></td>`;
            } else if (bid) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #666;">-</span></td>`;
            } else if (ask) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #666;">-</span></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }

            // Mid column
            if (mid) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><strong style="color: #0d47a1; font-size: 15px;">$${mid}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Vol column
            const vol = put.volume;
            if (typeof vol === 'number' && vol > 0) {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}">${vol.toLocaleString()}</td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // IV column
            const iv = put.implied_volatility;
            if (typeof iv === 'number') {
              html += `<td style="${cellStyle(putBg, 'right', true, true)}"><strong style="color: #1a1a1a;">${(iv * 100).toFixed(1)}%</strong></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Delta column
            const delta = put.delta;
            if (typeof delta === 'number') {
              html += `<td style="${cellStyle(putBg, 'center', true, true)}"><strong style="color: #004d40; font-size: 15px;">${delta.toFixed(3)}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }
            
            // Theta column
            const theta = put.theta;
            if (typeof theta === 'number') {
              html += `<td style="${cellStyle(putBg, 'center', true, false)}"><strong style="color: #bf360c; font-size: 15px;">${theta.toFixed(3)}</strong></td>`;
            } else {
              html += `<td style="${cellStyle(putBg, 'center', false, false)}"><span style="color: #666;">-</span></td>`;
            }
          } else {
            html += `<td colspan="6" style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
          }

          html += '</tr>';
        });

        html += '</tbody></table></div>';
      });

      optionsDisplay.innerHTML = html;
      debugLog(`Options rendered: ${sortedExpirations.length} expirations, ${contracts.length} total contracts`);
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
        newsHTML += `<li style="margin-bottom: 20px; padding: 20px; background: #ffffff; border-radius: 8px; border-left: 5px solid #1a73e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
          <h4 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 700; color: #1a1a1a; line-height: 1.4;">${title}</h4>
          <div style="font-size: 13px; color: #666; margin-bottom: 8px; font-weight: 500;">${published}</div>
          ${descSnippet ? `<p style="color: #333; line-height: 1.6; font-size: 15px; margin: 8px 0;">${descSnippet}</p>` : ''}
          <a href="${url}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: 600; font-size: 14px;">Read more →</a>
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
    debugLog('Chart data loaded, ready for initialization');
  }

  debugLog('Stock data rendered successfully');
  
  // Lazy-load options and news data if they weren't included in initial load
  const shouldLazyLoad = !stockData.options_info || !stockData.options_info.options_data;
  const shouldLazyLoadNews = !stockData.news_info || !stockData.news_info.news_data;
  
  if (shouldLazyLoad || shouldLazyLoadNews) {
    // Wait for page to be fully loaded before lazy-loading
    if (document.readyState === 'complete') {
      lazyLoadOptionsAndNews();
    } else {
      window.addEventListener('load', lazyLoadOptionsAndNews);
    }
  }
  
  async function lazyLoadOptionsAndNews() {
    const baseUrl = window.location.origin + window.location.pathname.split('/').slice(0, -1).join('/');
    
    // Lazy-load options
    if (shouldLazyLoad) {
      try {
        debugLog(`[${symbol}] Lazy-loading options data...`);
        const optionsUrl = `${baseUrl}/api/lazy/options/${symbol}`;
        const optionsResponse = await fetch(optionsUrl);
        if (optionsResponse.ok) {
          const optionsData = await optionsResponse.json();
          if (optionsData && optionsData.options_data) {
            // Update stockData with options
            stockData.options_info = optionsData;
            // Re-render options section
            renderOptionsSection(optionsData);
            debugLog(`[${symbol}] Options data lazy-loaded successfully`);
          }
        }
      } catch (e) {
        console.error(`[${symbol}] Error lazy-loading options:`, e);
      }
    }
    
    // Lazy-load news
    if (shouldLazyLoadNews) {
      try {
        debugLog(`[${symbol}] Lazy-loading news data...`);
        const newsUrl = `${baseUrl}/api/lazy/news/${symbol}`;
        const newsResponse = await fetch(newsUrl);
        if (newsResponse.ok) {
          const newsData = await newsResponse.json();
          if (newsData && newsData.news_data) {
            // Update stockData with news
            stockData.news_info = newsData;
            // Re-render news section
            renderNewsSection(newsData);
            debugLog(`[${symbol}] News data lazy-loaded successfully`);
          }
        }
      } catch (e) {
        console.error(`[${symbol}] Error lazy-loading news:`, e);
      }
    }
  }
  
  function renderOptionsSection(optionsInfo) {
    // Re-use the same options rendering logic from the main render function
    if (!optionsInfo || !optionsInfo.options_data) {
      return;
    }
    
    const optionsData = optionsInfo.options_data || {};
    debugLog(`[${symbol}] Lazy-load: Options data structure:`, {
      hasOptionsData: !!optionsData,
      hasSuccess: optionsData.success !== undefined,
      success: optionsData.success,
      hasData: !!optionsData.data,
      hasContracts: !!(optionsData.data && optionsData.data.contracts)
    });
    
    // Handle different options data structures
    let contracts = [];
    if (optionsData) {
      if (optionsData.success && optionsData.data && optionsData.data.contracts) {
        contracts = optionsData.data.contracts;
      } else if (Array.isArray(optionsData.contracts)) {
        contracts = optionsData.contracts;
      } else if (Array.isArray(optionsData)) {
        contracts = optionsData;
      }
    }
    
    if (contracts.length > 0) {
      const optionsSection = document.getElementById('optionsSection');
      const optionsDisplay = document.getElementById('optionsDisplay');
      if (optionsSection && optionsDisplay) {
        optionsSection.style.display = 'block';

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
        const mainPriceEl = document.getElementById('mainPrice');
        const currentPriceFromPage = mainPriceEl ? parseFloat(mainPriceEl.textContent.replace('$', '').replace(',', '')) : null;
        const currentPrice = currentPriceFromPage || 
                            parseFloat(currentPriceData.price || currentPriceData.close || currentPriceData.last_price || 0);

        // Build dropdown and tables (same logic as main render)
        let html = '<div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center;">';
        html += '<div><label for="optionsExpirationSelect" style="margin-right: 8px; font-weight: 600;">Expiration:</label>';
        html += '<select id="optionsExpirationSelect" onchange="showOptionsForExpiration(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">';
        sortedExpirations.forEach((exp, i) => {
          html += `<option value="${exp}" ${i === 0 ? 'selected' : ''}>${exp} (${byExpiry[exp].length} contracts)</option>`;
        });
        html += '</select></div>';
        
        html += '<div><label for="strikeRangeSelect" style="margin-right: 8px; font-weight: 600;">Show strikes:</label>';
        html += '<select id="strikeRangeSelect" onchange="filterStrikesByRange(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">';
        html += '<option value="10" selected>±10 around ATM</option>';
        html += '<option value="15">±15 around ATM</option>';
        html += '<option value="20">±20 around ATM</option>';
        html += '<option value="all">All strikes</option>';
        html += '</select></div></div>';

        // Build table for each expiration (same logic as main render, lines 854-1098)
        sortedExpirations.forEach((expDate, expIdx) => {
          const contractsList = byExpiry[expDate];
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
          
          let atmStrike = null;
          let atmStrikeIdx = null;
          if (currentPrice && sortedStrikes.length > 0) {
            atmStrikeIdx = sortedStrikes.reduce((minIdx, strike, idx) => {
              return Math.abs(strike - currentPrice) < Math.abs(sortedStrikes[minIdx] - currentPrice) ? idx : minIdx;
            }, 0);
            atmStrike = sortedStrikes[atmStrikeIdx];
          }

          let strikesToShow = sortedStrikes;
          if (currentPrice && atmStrike !== null) {
            const priceRange = 50;
            strikesToShow = sortedStrikes.filter(strike => Math.abs(strike - currentPrice) <= priceRange);
            if (strikesToShow.length < 50 && sortedStrikes.length > 50) {
              const startIdx = Math.max(0, atmStrikeIdx - 25);
              const endIdx = Math.min(sortedStrikes.length, atmStrikeIdx + 25);
              strikesToShow = sortedStrikes.slice(startIdx, endIdx);
            }
            if (strikesToShow.length === 0) {
              strikesToShow = sortedStrikes.slice(0, 100);
            }
          } else {
            strikesToShow = sortedStrikes.slice(0, 100);
          }
          
          const displayStyle = expIdx === 0 ? 'block' : 'none';
          html += `<div id="optionsTable_${expDate}" class="options-table-container" style="display: ${displayStyle}; margin-bottom: 20px;">`;
          html += '<table class="data-table options-chain-table" style="width: 100%; margin-bottom: 20px; font-size: 14px; border-collapse: separate; border-spacing: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; background: #ffffff;">';
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

          strikesToShow.forEach((strike, displayIdx) => {
            const originalIdx = sortedStrikes.indexOf(strike);
            const idx = originalIdx >= 0 ? originalIdx : displayIdx;
            const call = byStrike[strike].call;
            const put = byStrike[strike].put;

            const getRowBgColor = (strike, currentPrice, optionType, rowIdx) => {
              const baseColor = rowIdx % 2 === 0 ? '#ffffff' : '#f5f5f5';
              if (!currentPrice) return baseColor;
              const pctDiff = Math.abs(strike - currentPrice) / currentPrice * 100;
              if (pctDiff < 2) return '#e3f2fd';
              const isCall = optionType === 'call';
              const itm = isCall ? strike < currentPrice : strike > currentPrice;
              return itm ? '#fffbf0' : baseColor;
            };

            const distanceFromAtm = atmStrikeIdx !== null ? Math.abs(idx - atmStrikeIdx) : 999;
            const callBg = getRowBgColor(strike, currentPrice, 'call', idx);
            const putBg = getRowBgColor(strike, currentPrice, 'put', idx);

            const cellStyle = (bgColor, align = 'center', isNumeric = false, borderRight = false) => {
              let style = `padding: 12px 10px; background-color: ${bgColor}; text-align: ${align}; font-size: 14px; vertical-align: middle; color: #1a1a1a;`;
              if (borderRight) style += ' border-right: 1px solid #d0d0d0;';
              if (isNumeric) style += ' font-family: "SF Mono", "Monaco", "Courier New", monospace; font-weight: 600;';
              return style;
            };

            html += `<tr class="strike-row" data-distance-from-atm="${distanceFromAtm}" style="transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#e3f2fd'" onmouseout="this.style.backgroundColor=''">`;

            // Call columns
            if (call) {
              const bid = call.bid;
              const ask = call.ask;
              const mid = (bid && ask) ? ((bid + ask) / 2).toFixed(2) : null;
              const spread = (bid && ask) ? (ask - bid).toFixed(2) : null;

              if (bid && ask && bid > 0 && ask > 0) {
                html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #1b5e20; font-size: 13px;">$${spread}</strong></td>`;
              } else if (bid) {
                html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #666;">-</span></td>`;
              } else if (ask) {
                html += `<td style="${cellStyle(callBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #666;">-</span></td>`;
              } else {
                html += `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              }

              html += mid ? `<td style="${cellStyle(callBg, 'right', true, true)}"><strong style="color: #0d47a1; font-size: 15px;">$${mid}</strong></td>` : `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const vol = call.volume;
              html += (typeof vol === 'number' && vol > 0) ? `<td style="${cellStyle(callBg, 'right', true, true)}">${vol.toLocaleString()}</td>` : `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const iv = call.implied_volatility;
              html += (typeof iv === 'number') ? `<td style="${cellStyle(callBg, 'right', true, true)}"><strong style="color: #1a1a1a;">${(iv * 100).toFixed(1)}%</strong></td>` : `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const delta = call.delta;
              html += (typeof delta === 'number') ? `<td style="${cellStyle(callBg, 'center', true, true)}"><strong style="color: #004d40; font-size: 15px;">${delta.toFixed(3)}</strong></td>` : `<td style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const theta = call.theta;
              html += (typeof theta === 'number') ? `<td style="${cellStyle(callBg, 'center', true, false)}"><strong style="color: #bf360c; font-size: 15px;">${theta.toFixed(3)}</strong></td>` : `<td style="${cellStyle(callBg, 'center', false, false)}"><span style="color: #666;">-</span></td>`;
            } else {
              html += `<td colspan="6" style="${cellStyle(callBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }

            html += `<td style="padding: 12px 10px; background-color: #5e72e4; color: white; text-align: center; font-weight: 700; font-size: 14px; border-left: 2px solid white; border-right: 2px solid white;">$${strike.toFixed(2)}</td>`;

            // Put columns
            if (put) {
              const bid = put.bid;
              const ask = put.ask;
              const mid = (bid && ask) ? ((bid + ask) / 2).toFixed(2) : null;
              const spread = (bid && ask) ? (ask - bid).toFixed(2) : null;

              if (bid && ask && bid > 0 && ask > 0) {
                html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / $${ask.toFixed(2)}</span><br><strong style="color: #b71c1c; font-size: 13px;">$${spread}</strong></td>`;
              } else if (bid) {
                html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">$${bid.toFixed(2)} / -</span><br><span style="color: #666;">-</span></td>`;
              } else if (ask) {
                html += `<td style="${cellStyle(putBg, 'right', true, true)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / $${ask.toFixed(2)}</span><br><span style="color: #666;">-</span></td>`;
              } else {
                html += `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              }

              html += mid ? `<td style="${cellStyle(putBg, 'right', true, true)}"><strong style="color: #0d47a1; font-size: 15px;">$${mid}</strong></td>` : `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const vol = put.volume;
              html += (typeof vol === 'number' && vol > 0) ? `<td style="${cellStyle(putBg, 'right', true, true)}">${vol.toLocaleString()}</td>` : `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const iv = put.implied_volatility;
              html += (typeof iv === 'number') ? `<td style="${cellStyle(putBg, 'right', true, true)}"><strong style="color: #1a1a1a;">${(iv * 100).toFixed(1)}%</strong></td>` : `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const delta = put.delta;
              html += (typeof delta === 'number') ? `<td style="${cellStyle(putBg, 'center', true, true)}"><strong style="color: #004d40; font-size: 15px;">${delta.toFixed(3)}</strong></td>` : `<td style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
              const theta = put.theta;
              html += (typeof theta === 'number') ? `<td style="${cellStyle(putBg, 'center', true, false)}"><strong style="color: #bf360c; font-size: 15px;">${theta.toFixed(3)}</strong></td>` : `<td style="${cellStyle(putBg, 'center', false, false)}"><span style="color: #666;">-</span></td>`;
            } else {
              html += `<td colspan="6" style="${cellStyle(putBg, 'center', false, true)}"><span style="color: #666;">-</span></td>`;
            }

            html += '</tr>';
          });

          html += '</tbody></table></div>';
        });

        optionsDisplay.innerHTML = html;
        debugLog(`[${symbol}] Options rendered: ${sortedExpirations.length} expirations, ${contracts.length} total contracts`);
      }
    } else if (optionsInfo && optionsInfo.error) {
      const optionsSection = document.getElementById('optionsSection');
      const optionsDisplay = document.getElementById('optionsDisplay');
      if (optionsSection && optionsDisplay) {
        optionsSection.style.display = 'block';
        optionsDisplay.innerHTML = `<p style="color: #f85149;">Error loading options: ${optionsInfo.error}</p>`;
      }
    }
  }
  
  function renderNewsSection(newsInfo) {
    // Update stockData and re-run news rendering
    if (newsInfo && newsInfo.news_data) {
      stockData.news_info = newsInfo;
      const newsData = newsInfo.news_data || {};
      const newsArticles = newsData.articles || [];
      
      if (newsArticles.length > 0) {
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
            newsHTML += `<li style="margin-bottom: 20px; padding: 20px; background: #ffffff; border-radius: 8px; border-left: 5px solid #1a73e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
              <h4 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 700; color: #1a1a1a; line-height: 1.4;">${title}</h4>
              <div style="font-size: 13px; color: #666; margin-bottom: 8px; font-weight: 500;">${published}</div>
              ${descSnippet ? `<p style="color: #333; line-height: 1.6; font-size: 15px; margin: 8px 0;">${descSnippet}</p>` : ''}
              <a href="${url}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: 600; font-size: 14px;">Read more →</a>
            </li>`;
          });
          newsHTML += '</ul>';
          newsDisplay.innerHTML = newsHTML;
          debugLog(`[${symbol}] News section rendered with ${newsArticles.length} articles`);
        }
      }
    }
  }
})();

// Options expiration and strike filtering functions (must be global)
function showOptionsForExpiration(expiration) {
  // Hide all options tables
  const allTables = document.querySelectorAll('.options-table-container');
  allTables.forEach(table => {
    table.style.display = 'none';
  });
  
  // Show the selected table
  const selectedTable = document.getElementById('optionsTable_' + expiration);
  if (selectedTable) {
    selectedTable.style.display = 'block';
    // Re-apply strike range filter
    const rangeSelect = document.getElementById('strikeRangeSelect');
    if (rangeSelect) {
      filterStrikesByRange(rangeSelect.value);
    }
  }
}

function filterStrikesByRange(range) {
  // Get all visible options tables
  const visibleTable = document.querySelector('.options-table-container[style*="display: block"]');
  if (!visibleTable) return;
  
  const rows = visibleTable.querySelectorAll('.strike-row');
  
  if (range === 'all') {
    // Show all rows
    rows.forEach(row => {
      row.style.display = '';
    });
  } else {
    const maxStrikes = parseInt(range); // e.g., 10 means show ±10 strikes around ATM
    rows.forEach(row => {
      // Get distance from ATM from the data attribute (set when row was created)
      const distance = parseInt(row.getAttribute('data-distance-from-atm') || '999');
      if (distance <= maxStrikes) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }
}

// Apply default strike range filter on load
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  setTimeout(() => filterStrikesByRange('10'), 100);
} else {
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => filterStrikesByRange('10'), 100);
  });
}

