"""
Streak Analyzer Module

This module provides functionality to analyze price streaks in stock data.
It can be imported and used by other programs for strategy testing and analysis.
"""

import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional


class StreakAnalyzer:
    """
    A class to analyze price streaks in stock data.
    
    This class provides methods to compute up/down streaks, analyze their patterns,
    and generate statistics that can be used for strategy testing.
    """
    
    def __init__(self):
        pass
    
    def compute_streaks(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Compute up and down streaks from price data.
        
        Args:
            df: DataFrame with price data, must have 'close' column and datetime index
            
        Returns:
            Tuple of (up_streaks, down_streaks) where each is a list of dictionaries
            containing streak information:
            - start_date: start date of streak
            - end_date: end date of streak  
            - length: number of days in streak
            - avg_movement: total percentage movement from start to end
            - start_price: price at start of streak
            - end_price: price at end of streak
        """
        if df.empty or 'close' not in df.columns:
            return [], []
            
        df = df.copy()
        df = df.sort_index()
        
        closes = df['close'].values
        dates = df.index.to_list()
        up_streaks = []
        down_streaks = []
        streak_type = None
        streak_start = 0
        streak_len = 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                if streak_type == 'up':
                    streak_len += 1
                else:
                    if streak_type == 'down' and streak_len > 0:
                        # End of down streak
                        start_price = closes[streak_start]
                        end_price = closes[i-1]
                        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                        down_streaks.append({
                            'start_date': dates[streak_start],
                            'end_date': dates[i-1],
                            'length': streak_len,
                            'avg_movement': total_movement,
                            'start_price': start_price,
                            'end_price': end_price
                        })
                    streak_type = 'up'
                    streak_start = i-1
                    streak_len = 1
            elif closes[i] < closes[i-1]:
                if streak_type == 'down':
                    streak_len += 1
                else:
                    if streak_type == 'up' and streak_len > 0:
                        # End of up streak
                        start_price = closes[streak_start]
                        end_price = closes[i-1]
                        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                        up_streaks.append({
                            'start_date': dates[streak_start],
                            'end_date': dates[i-1],
                            'length': streak_len,
                            'avg_movement': total_movement,
                            'start_price': start_price,
                            'end_price': end_price
                        })
                    streak_type = 'down'
                    streak_start = i-1
                    streak_len = 1
            else:
                # Flat day, treat as continuation only if a streak is ongoing
                if streak_type in ('up', 'down'):
                    streak_len += 1
                # else: do nothing, just move past the day
        
        # Add last streak
        if streak_type == 'up' and streak_len > 0:
            start_price = closes[streak_start]
            end_price = closes[len(closes)-1]
            total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            up_streaks.append({
                'start_date': dates[streak_start],
                'end_date': dates[len(closes)-1],
                'length': streak_len,
                'avg_movement': total_movement,
                'start_price': start_price,
                'end_price': end_price
            })
        elif streak_type == 'down' and streak_len > 0:
            start_price = closes[streak_start]
            end_price = closes[len(closes)-1]
            total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            down_streaks.append({
                'start_date': dates[streak_start],
                'end_date': dates[len(closes)-1],
                'length': streak_len,
                'avg_movement': total_movement,
                'start_price': start_price,
                'end_price': end_price
            })
        
        return up_streaks, down_streaks
    
    def analyze_hourly_streaks(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze hourly streaks (consecutive up/down hours).
        
        Args:
            df: DataFrame with hourly price data
            
        Returns:
            Tuple of (up_streaks, down_streaks) with hourly streak information
        """
        if df.empty or 'close' not in df.columns:
            return [], []
            
        df = df.copy()
        df = df.sort_index()
        
        closes = df['close'].values
        dates = df.index.to_list()
        up_streaks = []
        down_streaks = []
        streak_type = None
        streak_start = 0
        streak_len = 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                if streak_type == 'up':
                    streak_len += 1
                else:
                    if streak_type == 'down' and streak_len > 0:
                        start_price = closes[streak_start]
                        end_price = closes[i-1]
                        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                        down_streaks.append({
                            'start_date': dates[streak_start],
                            'end_date': dates[i-1],
                            'length': streak_len,
                            'avg_movement': total_movement,
                            'start_price': start_price,
                            'end_price': end_price
                        })
                    streak_type = 'up'
                    streak_start = i-1
                    streak_len = 1
            elif closes[i] < closes[i-1]:
                if streak_type == 'down':
                    streak_len += 1
                else:
                    if streak_type == 'up' and streak_len > 0:
                        start_price = closes[streak_start]
                        end_price = closes[i-1]
                        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                        up_streaks.append({
                            'start_date': dates[streak_start],
                            'end_date': dates[i-1],
                            'length': streak_len,
                            'avg_movement': total_movement,
                            'start_price': start_price,
                            'end_price': end_price
                        })
                    streak_type = 'down'
                    streak_start = i-1
                    streak_len = 1
            else:
                # Flat hour, treat as continuation only if a streak is ongoing
                if streak_type in ('up', 'down'):
                    streak_len += 1
                # else: do nothing, just move past the hour
        
        # Add last streak
        if streak_type == 'up' and streak_len > 0:
            start_price = closes[streak_start]
            end_price = closes[len(closes)-1]
            total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            up_streaks.append({
                'start_date': dates[streak_start],
                'end_date': dates[len(closes)-1],
                'length': streak_len,
                'avg_movement': total_movement,
                'start_price': start_price,
                'end_price': end_price
            })
        elif streak_type == 'down' and streak_len > 0:
            start_price = closes[streak_start]
            end_price = closes[len(closes)-1]
            total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            down_streaks.append({
                'start_date': dates[streak_start],
                'end_date': dates[len(closes)-1],
                'length': streak_len,
                'avg_movement': total_movement,
                'start_price': start_price,
                'end_price': end_price
            })
        
        return up_streaks, down_streaks
    
    def get_streak_statistics(self, up_streaks: List[Dict], down_streaks: List[Dict]) -> Dict:
        """
        Get comprehensive statistics about streaks.
        
        Args:
            up_streaks: List of up streak dictionaries
            down_streaks: List of down streak dictionaries
            
        Returns:
            Dictionary containing streak statistics
        """
        total_up_days = sum(s['length'] for s in up_streaks)
        total_down_days = sum(s['length'] for s in down_streaks)
        total_streak_days = total_up_days + total_down_days
        
        # Length frequency analysis
        up_lengths = [s['length'] for s in up_streaks]
        down_lengths = [s['length'] for s in down_streaks]
        
        up_length_freq = Counter(up_lengths)
        down_length_freq = Counter(down_lengths)
        
        # Movement analysis
        up_movements = [s['avg_movement'] for s in up_streaks]
        down_movements = [s['avg_movement'] for s in down_streaks]
        
        stats = {
            'total_up_days': total_up_days,
            'total_down_days': total_down_days,
            'total_streak_days': total_streak_days,
            'up_streak_count': len(up_streaks),
            'down_streak_count': len(down_streaks),
            'up_length_frequency': dict(up_length_freq),
            'down_length_frequency': dict(down_length_freq),
            'up_movements': up_movements,
            'down_movements': down_movements,
            'avg_up_movement': sum(up_movements) / len(up_movements) if up_movements else 0,
            'avg_down_movement': sum(down_movements) / len(down_movements) if down_movements else 0,
            'max_up_streak_length': max(up_lengths) if up_lengths else 0,
            'max_down_streak_length': max(down_lengths) if down_lengths else 0,
            'min_up_streak_length': min(up_lengths) if up_lengths else 0,
            'min_down_streak_length': min(down_lengths) if down_lengths else 0,
        }
        
        return stats
    
    def filter_streaks_by_length(self, streaks: List[Dict], min_length: int = 1, max_length: Optional[int] = None) -> List[Dict]:
        """
        Filter streaks by length criteria.
        
        Args:
            streaks: List of streak dictionaries
            min_length: Minimum streak length to include
            max_length: Maximum streak length to include (None for no limit)
            
        Returns:
            Filtered list of streaks
        """
        if max_length is None:
            return [s for s in streaks if s['length'] >= min_length]
        else:
            return [s for s in streaks if min_length <= s['length'] <= max_length]
    
    def filter_streaks_by_movement(self, streaks: List[Dict], min_movement: float = 0.0, max_movement: Optional[float] = None) -> List[Dict]:
        """
        Filter streaks by movement criteria.
        
        Args:
            streaks: List of streak dictionaries
            min_movement: Minimum movement percentage to include
            max_movement: Maximum movement percentage to include (None for no limit)
            
        Returns:
            Filtered list of streaks
        """
        if max_movement is None:
            return [s for s in streaks if abs(s['avg_movement']) >= min_movement]
        else:
            return [s for s in streaks if min_movement <= abs(s['avg_movement']) <= max_movement]
    
    def get_streaks_in_date_range(self, streaks: List[Dict], start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict]:
        """
        Get streaks that fall within a specific date range.
        
        Args:
            streaks: List of streak dictionaries
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            List of streaks within the date range
        """
        return [s for s in streaks if start_date <= s['start_date'] <= end_date or start_date <= s['end_date'] <= end_date]
    
    def get_current_streak(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Get the current ongoing streak (if any).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with current streak info or None if no current streak
        """
        if df.empty or len(df) < 2:
            return None
            
        df = df.copy()
        df = df.sort_index()
        
        # Get the last few data points to determine current streak
        last_closes = df['close'].tail(10).values  # Look at last 10 points
        
        if len(last_closes) < 2:
            return None
            
        # Determine streak direction
        streak_type = None
        streak_length = 0
        
        for i in range(len(last_closes) - 1, 0, -1):
            if last_closes[i] > last_closes[i-1]:
                if streak_type == 'up' or streak_type is None:
                    streak_type = 'up'
                    streak_length += 1
                else:
                    break
            elif last_closes[i] < last_closes[i-1]:
                if streak_type == 'down' or streak_type is None:
                    streak_type = 'down'
                    streak_length += 1
                else:
                    break
            else:
                # Flat, continue streak
                streak_length += 1
        
        if streak_length > 1:
            start_idx = len(df) - streak_length
            start_price = df.iloc[start_idx]['close']
            end_price = df.iloc[-1]['close']
            total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            
            return {
                'type': streak_type,
                'length': streak_length,
                'start_date': df.index[start_idx],
                'end_date': df.index[-1],
                'start_price': start_price,
                'end_price': end_price,
                'avg_movement': total_movement,
                'is_current': True
            }
        
        return None


# Convenience functions for backward compatibility
def compute_streaks(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Convenience function to compute streaks using the default analyzer."""
    analyzer = StreakAnalyzer()
    return analyzer.compute_streaks(df)


def analyze_hourly_streaks(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Convenience function to analyze hourly streaks using the default analyzer."""
    analyzer = StreakAnalyzer()
    return analyzer.analyze_hourly_streaks(df)
