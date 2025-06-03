import threading
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import heapq
from typing import Dict, Optional

class StaticDisplayManager:
    def __init__(self, all_symbols: list[str], output_handle=sys.stdout, display_update_interval: float = 1.0):
        self.lock = threading.Lock()
        self.symbols = sorted(list(set(all_symbols)))
        self.symbol_to_row_offset = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.num_display_lines = len(self.symbols)
        self.display_prepared = False
        self.output_handle = output_handle
        self.header_lines = 1
        self.footer_lines = 1
        self.display_update_interval = display_update_interval
        self.latest_data_buffer = {symbol: "Waiting for data..." for symbol in self.symbols}
        self.updater_thread = None
        self.stop_event = threading.Event()
        
        # Force debug output to stderr
        print(f"DEBUG: StaticDisplayManager initialized with {len(self.symbols)} symbols", file=sys.stderr, flush=True)
        print(f"DEBUG: Symbols: {self.symbols}", file=sys.stderr, flush=True)

    def _print(self, *args, **kwargs):
        try:
            # Ensure we're writing to stdout for display
            kwargs['file'] = self.output_handle
            kwargs['flush'] = True
            print(*args, **kwargs)
        except Exception as e:
            print(f"DEBUG: Error in _print: {e}", file=sys.stderr, flush=True)

    def prepare_display(self):
        print("DEBUG: Preparing display", file=sys.stderr, flush=True)
        with self.lock:
            if self.display_prepared:
                print("DEBUG: Display already prepared", file=sys.stderr, flush=True)
                return
            if self.num_display_lines == 0:
                print("DEBUG: No lines to display", file=sys.stderr, flush=True)
                return

            try:
                print("DEBUG: Starting display preparation", file=sys.stderr, flush=True)
                
                # Clear screen and move to top
                self._print("\x1b[2J\x1b[H", end="")
                
                # Print header
                self._print("=== Real-time Market Updates ===")
                self._print("Symbol          Bid Price        Ask Price        Time")
                self._print("-" * 60)
                
                # Print initial state for each symbol
                for symbol in self.symbols:
                    self._print(f"{symbol:<15}: {self.latest_data_buffer[symbol]}")
                
                # Print footer
                self._print("-" * 60)
                
                self.display_prepared = True
                print("DEBUG: Display prepared successfully", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"DEBUG: Error in prepare_display: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
        
        if self.num_display_lines > 0 and self.display_update_interval > 0:
            print("DEBUG: Starting display updater thread", file=sys.stderr, flush=True)
            self.stop_event.clear()
            self.updater_thread = threading.Thread(target=self._updater_thread_target, daemon=True)
            self.updater_thread.start()
            print("DEBUG: Display updater thread started", file=sys.stderr, flush=True)

    def _updater_thread_target(self):
        print("DEBUG: Starting display updater thread", file=sys.stderr, flush=True)
        while not self.stop_event.is_set():
            try:
                time.sleep(self.display_update_interval)
                with self.lock:
                    if not self.display_prepared:
                        print("DEBUG: Display not prepared yet", file=sys.stderr, flush=True)
                        continue
                    if self.num_display_lines == 0:
                        print("DEBUG: No lines to display", file=sys.stderr, flush=True)
                        continue

                    # Clear screen and move to top
                    self._print("\x1b[2J\x1b[H", end="")
                    
                    # Print header
                    self._print("=== Real-time Market Updates ===")
                    self._print("Symbol          Bid Price        Ask Price        Time")
                    self._print("-" * 60)
                    
                    # Print each symbol's data
                    for symbol in self.symbols:
                        data_str = self.latest_data_buffer.get(symbol, "Error: No data")
                        self._print(f"{symbol:<15}: {data_str}")
                    
                    # Print footer
                    self._print("-" * 60)
            except Exception as e:
                print(f"DEBUG: Error in _updater_thread_target: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

    def update_symbol(self, symbol: str, data_str: str):
        try:
            with self.lock:
                if symbol in self.latest_data_buffer:
                    self.latest_data_buffer[symbol] = data_str
                else:
                    print(f"DEBUG: Symbol {symbol} not found in buffer", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"DEBUG: Error in update_symbol: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    def cleanup_display(self):
        print("DEBUG: Cleaning up display", file=sys.stderr, flush=True)
        if self.updater_thread and self.updater_thread.is_alive():
            self.stop_event.set()
            self.updater_thread.join(timeout=self.display_update_interval * 2)

        with self.lock:
            if not self.display_prepared or self.num_display_lines == 0:
                return
            
            self._print("\n" * 2)
            self._print("-" * 40)
            self._print("Static display ended.")
            self._print("\n" * 2)
            self.display_prepared = False
            print("DEBUG: Display cleanup complete", file=sys.stderr, flush=True)
        return 0

# --- End Static Display Manager ---
class ActivityTracker:
    """Tracks stock activity over a time window."""

    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.activity_data = defaultdict(
            list
        )  # symbol -> list of (timestamp, ask_size) tuples
        self.lock = threading.Lock()
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_old_data, daemon=True
        )
        self.cleanup_thread.start()
        print(f"DEBUG: ActivityTracker initialized with {window_seconds}s window", file=sys.stderr, flush=True)

    def add_activity(self, symbol: str, timestamp: datetime, ask_size: int):
        """Add a new activity data point."""
        with self.lock:
            self.activity_data[symbol].append((timestamp, ask_size))

    def _cleanup_old_data(self):
        """Periodically remove old data points."""
        while True:
            time.sleep(1)  # Check every second
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(seconds=self.window_seconds)

            with self.lock:
                for symbol in list(self.activity_data.keys()):
                    # Keep only data points within the window
                    self.activity_data[symbol] = [
                        (ts, size)
                        for ts, size in self.activity_data[symbol]
                        if ts > cutoff_time
                    ]
                    # Remove empty lists
                    if not self.activity_data[symbol]:
                        del self.activity_data[symbol]

    def get_most_active_symbols(self, n: int) -> list[str]:
        """Get the n most active symbols based on total ask size in the window."""
        with self.lock:
            # Calculate total ask size for each symbol
            symbol_totals = {
                symbol: sum(size for _, size in data)
                for symbol, data in self.activity_data.items()
            }

            # Get top n symbols using a min heap
            if len(symbol_totals) <= n:
                return list(symbol_totals.keys())

            # Use negative size for max heap behavior
            heap = [(-size, symbol) for symbol, size in symbol_totals.items()]
            heapq.heapify(heap)

            # Get top n symbols
            return [heapq.heappop(heap)[1] for _ in range(n)]


class DynamicDisplayManager(StaticDisplayManager):
    """Extends StaticDisplayManager to handle dynamic symbol updates."""

    def __init__(
        self, 
        activity_tracker: ActivityTracker, 
        max_symbols: int, 
        initial_symbols: list[str],
        output_handle=sys.stdout,
        display_update_interval: float = 1.0
    ):
        super().__init__(initial_symbols, output_handle, display_update_interval)
        self.activity_tracker = activity_tracker
        self.max_symbols = max_symbols
        self.update_thread = threading.Thread(
            target=self._update_symbols_thread, daemon=True
        )
        self.stop_event = threading.Event()
        self.display_lock = threading.Lock()  # Separate lock for display updates
        self.last_prices: Dict[str, Dict[str, Optional[float]]] = {}
        print(f"DEBUG: DynamicDisplayManager initialized with max_symbols={max_symbols}", file=sys.stderr, flush=True)

    def _update_symbols_thread(self):
        """Periodically updates the list of displayed symbols."""
        print("DEBUG: Starting symbol update thread", file=sys.stderr, flush=True)
        while not self.stop_event.is_set():
            try:
                new_symbols = self.activity_tracker.get_most_active_symbols(self.max_symbols)
                if not new_symbols:
                    # If no active symbols yet, use initial symbols
                    new_symbols = self.symbols
                
                with self.display_lock:
                    # Only update if the symbols have changed
                    if set(new_symbols) != set(self.symbols):
                        print(f"DEBUG: Updating displayed symbols to: {new_symbols}", file=sys.stderr, flush=True)
                        self.symbols = sorted(new_symbols)
                        self.symbol_to_row_offset = {symbol: i for i, symbol in enumerate(self.symbols)}
                        self.num_display_lines = len(self.symbols)
                        
                        # Reset display
                        self.display_prepared = False
                        self.prepare_display()
                
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"DEBUG: Error in symbol update thread: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

    def _updater_thread_target(self):
        """Periodically updates the display with buffered data."""
        print("DEBUG: Starting display updater thread", file=sys.stderr, flush=True)
        while not self.stop_event.is_set():
            try:
                time.sleep(self.display_update_interval)
                with self.display_lock:
                    if not self.display_prepared:
                        print("DEBUG: Display not prepared yet", file=sys.stderr, flush=True)
                        continue
                    if self.num_display_lines == 0:
                        print("DEBUG: No lines to display", file=sys.stderr, flush=True)
                        continue

                    # Clear screen and move to top
                    self._print("\x1b[2J\x1b[H", end="")
                    
                    # Print header
                    self._print("=== Real-time Market Updates ===")
                    self._print("Symbol          Bid Price        Ask Price        Time")
                    self._print("-" * 60)
                    
                    # Print each symbol's data
                    for symbol in self.symbols:
                        data_str = self.latest_data_buffer.get(symbol, "Waiting for data...")
                        self._print(f"{symbol:<15}: {data_str}")
                    
                    # Print footer
                    self._print("-" * 60)
            except Exception as e:
                print(f"DEBUG: Error in display updater thread: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

    def update_symbol(self, symbol: str, data_str: str):
        """Update a symbol's data in the buffer."""
        try:
            with self.lock:
                if symbol in self.latest_data_buffer:
                    self.latest_data_buffer[symbol] = data_str
                else:
                    print(f"DEBUG: Symbol {symbol} not found in buffer", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"DEBUG: Error in update_symbol: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    def start(self):
        """Start the dynamic display manager."""
        print("DEBUG: Starting DynamicDisplayManager", file=sys.stderr, flush=True)
        self.stop_event.clear()
        self.update_thread.start()
        # Also start the display updater thread
        super().prepare_display()

    def cleanup_display(self):
        """Clean up the display and stop the update thread."""
        print("DEBUG: Cleaning up DynamicDisplayManager", file=sys.stderr, flush=True)
        self.stop_event.set()
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        super().cleanup_display()
