"""
Route registration for db_server.

Centralizes all route registration to avoid duplication and ensure consistency.
"""

from aiohttp import web


def register_routes(app: web.Application) -> None:
    """
    Register all API routes with the application.
    
    This function should be called once during application setup.
    Routes are registered in order, with more specific routes before parameterized ones.
    
    Args:
        app: aiohttp Application instance
        
    Note:
        Route order matters! More specific routes must be registered before
        parameterized routes (e.g., /stock_info/ws before /stock_info/{symbol})
    """
    # Import handlers here to avoid circular imports
    # Handlers will be imported from db_server.py initially, then migrated to handler modules
    
    # Import from db_server for now (will be migrated to handler modules)
    from db_server import (
        handle_run_script,
        handle_db_command,
        handle_websocket,
        handle_health_check,
        handle_stats_database,
        handle_stats_tables,
        handle_stats_performance,
        handle_stats_pool,
        handle_stats_redis,
        handle_analyze_ticker,
        handle_stock_info,
        handle_stock_analysis,
        handle_ai_query,
        handle_execute_sql,
        handle_yahoo_finance_news,
        handle_twitter_tweets,
        handle_reddit_news,
        handle_wsb_daily_thread,
        handle_covered_calls_data,
        handle_covered_calls_analysis,
        handle_covered_calls_view,
        handle_covered_calls_static,
        handle_stock_analysis_data,
        handle_lazy_load_options,
        handle_lazy_load_news,
        handle_lazy_load_chart,
        handle_lazy_load_strategies,
        handle_stock_info_static,
        handle_stock_info_html,
        handle_predictions_page,
        handle_lazy_load_today_prediction,
        handle_lazy_load_future_prediction,
        handle_lazy_load_band_history,
        handle_lazy_load_historical_prediction,
        handle_prewarm_predictions,
        handle_range_percentiles_api,
        handle_range_percentiles_html,
        handle_range_percentiles_multi_window_api,
        handle_catch_all,
    )
    
    # Run script endpoint (run_scripts/ with ?script=...&start_date=...&end_date=...)
    app.router.add_get("/run_script", handle_run_script)

    # Core endpoints
    app.router.add_post("/db_command", handle_db_command)
    app.router.add_get("/ws", handle_websocket)
    app.router.add_get("/", handle_health_check)
    app.router.add_get("/health", handle_health_check)
    
    # Stats endpoints
    app.router.add_get("/stats/database", handle_stats_database)
    app.router.add_get("/stats/tables", handle_stats_tables)
    app.router.add_get("/stats/performance", handle_stats_performance)
    app.router.add_get("/stats/pool", handle_stats_pool)
    app.router.add_get("/stats/redis", handle_stats_redis)
    
    # Ticker analysis endpoint
    app.router.add_get("/analyze_ticker", handle_analyze_ticker)
    app.router.add_post("/analyze_ticker", handle_analyze_ticker)
    
    # Stock info API endpoint
    app.router.add_get("/api/stock_info/{symbol}", handle_stock_info)
    
    # Stock analysis API endpoint
    app.router.add_get("/api/stock_analysis", handle_stock_analysis)
    
    # AI query API endpoint
    app.router.add_get("/api/ai_query", handle_ai_query)
    
    # SQL execution API endpoint
    app.router.add_get("/api/execute_sql", handle_execute_sql)
    app.router.add_get("/api/sql_query", handle_execute_sql)  # Alias for execute_sql
    
    # News endpoints
    app.router.add_get("/api/yahoo_news/{symbol}", handle_yahoo_finance_news)
    app.router.add_get("/api/tweets/{symbol}", handle_twitter_tweets)
    app.router.add_get("/api/reddit_news/{symbol}", handle_reddit_news)
    app.router.add_get("/api/wsb_daily_thread", handle_wsb_daily_thread)
    
    # Stock info API subroutes BEFORE the parameterized route
    # (must be registered before /stock_info/{symbol} to avoid {symbol} capturing "ws" or "api")
    app.router.add_get("/stock_info/ws", handle_websocket)
    app.router.add_get("/stock_info/api/covered_calls/data", handle_covered_calls_data)
    app.router.add_get("/stock_info/api/covered_calls/analysis", handle_covered_calls_analysis)
    app.router.add_get("/stock_info/api/covered_calls/view", handle_covered_calls_view)
    app.router.add_get("/stock_info/api/covered_calls/{filename}", handle_covered_calls_static)
    app.router.add_get("/stock_info/api/stock_analysis/data", handle_stock_analysis_data)
    app.router.add_get("/stock_info/api/lazy/options/{symbol}", handle_lazy_load_options)
    app.router.add_get("/stock_info/api/lazy/news/{symbol}", handle_lazy_load_news)
    app.router.add_get("/stock_info/api/lazy/chart/{symbol}", handle_lazy_load_chart)
    app.router.add_get("/stock_info/api/lazy/strategies/{symbol}", handle_lazy_load_strategies)
    app.router.add_get("/static/stock_info/{filename}", handle_stock_info_static)
    
    # Stock info HTML page endpoint (parameterized route must be after specific routes)
    app.router.add_get("/stock_info/{symbol}", handle_stock_info_html)

    # Prediction endpoints (must be before catch-all)
    app.router.add_get("/predictions/api/prewarm", handle_prewarm_predictions)
    app.router.add_get("/predictions/api/lazy/today/{ticker}", handle_lazy_load_today_prediction)
    app.router.add_get("/predictions/api/lazy/future/{ticker}/{days}", handle_lazy_load_future_prediction)
    app.router.add_get("/predictions/api/lazy/band_history/{ticker}", handle_lazy_load_band_history)
    app.router.add_get("/predictions/api/lazy/historical/{ticker}/{date}", handle_lazy_load_historical_prediction)
    app.router.add_get("/predictions/{ticker}", handle_predictions_page)

    # Range percentiles endpoints
    app.router.add_get("/api/range_percentiles", handle_range_percentiles_api)
    app.router.add_get("/api/range_percentiles_multi_window", handle_range_percentiles_multi_window_api)
    app.router.add_get("/range_percentiles", handle_range_percentiles_html)

    # Catch-all handler for unknown routes (must be last)
    app.router.add_get("/{path:.*}", handle_catch_all)
    app.router.add_post("/{path:.*}", handle_catch_all)
