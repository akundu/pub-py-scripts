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
  const currentPrice = currentPriceData.price || currentPriceData.close || currentPriceData.last_price;
  const previousClose = currentPriceData.previous_close || financialInfo.financial_data?.previous_close;
  const priceChange = currentPriceData.change || currentPriceData.change_amount || 0;
  const priceChangePct = currentPriceData.change_percent || currentPriceData.change_pct || 0;

  if (currentPrice) {
    document.getElementById('mainPrice').textContent = '$' + parseFloat(currentPrice).toFixed(2);
  }

  const changeElement = document.getElementById('mainChange');
  const changeSign = priceChange >= 0 ? '+' : '';
  const changeColor = priceChange >= 0 ? 'positive' : 'negative';
  changeElement.textContent = `${changeSign}$${Math.abs(priceChange).toFixed(2)} (${changeSign}${priceChangePct.toFixed(2)}%)`;
  changeElement.className = 'change ' + changeColor;

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
      { label: 'EPS (TTM)', value: formatValue(financialData.earnings_per_share || financialData.eps) },
      { label: 'PE Ratio (TTM)', value: formatValue(financialData.price_to_earnings || financialData.pe_ratio) },
      { label: 'PEG Ratio', value: formatValue(financialData.peg_ratio) },
      { label: 'Return on Equity (ROE)', value: formatValue(financialData.return_on_equity, false, true) },
      { label: 'Return on Assets (ROA)', value: formatValue(financialData.return_on_assets, false, true) },
      { label: 'Price to Book (P/B)', value: formatValue(financialData.price_to_book) },
      { label: 'Price to Sales (P/S)', value: formatValue(financialData.price_to_sales) },
      { label: 'Price to Cash Flow (P/CF)', value: formatValue(financialData.price_to_cash_flow) },
      { label: 'Price to Free Cash Flow (P/FCF)', value: formatValue(financialData.price_to_free_cash_flow) },
      { label: 'EV to Sales', value: formatValue(financialData.ev_to_sales) },
      { label: 'EV to EBITDA', value: formatValue(financialData.ev_to_ebitda) },
      { label: 'Enterprise Value', value: formatValue(financialData.enterprise_value, true) },
      { label: 'Free Cash Flow', value: formatValue(financialData.free_cash_flow, true) },
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

  // Trigger chart initialization if the chart script is loaded
  if (typeof Chart !== 'undefined' && document.getElementById('priceChart')) {
    // Chart initialization will be handled by existing code
    console.log('Chart data loaded, ready for initialization');
  }

  console.log('Stock data rendered successfully');
})();

