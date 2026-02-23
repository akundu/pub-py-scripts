#!/usr/bin/env python3
"""
Monitor prediction model health and alert if retraining is needed.

This script checks:
1. Days since last retrain
2. Recent prediction accuracy (if available)
3. Model file integrity

Usage:
    python scripts/monitor_model_health.py
    python scripts/monitor_model_health.py --ticker NDX
    python scripts/monitor_model_health.py --ticker SPX --alert-email your@email.com
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


class ModelHealthMonitor:
    """Monitor prediction model health."""

    def __init__(self, project_dir: Path, ticker: str = "NDX"):
        self.project_dir = project_dir
        self.ticker = ticker
        self.prod_dir = project_dir / "models" / "production" / ticker
        self.results_dir = project_dir / "results"

    def check_model_age(self) -> dict:
        """Check days since last retraining."""

        metadata_file = self.prod_dir / "metadata.json"

        if not metadata_file.exists():
            return {
                'status': 'error',
                'message': 'No production models found',
                'days_old': None,
                'should_retrain': True,
            }

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            retrained_at = metadata.get('retrained_date', metadata.get('retrained_at', ''))[:8]
            retrain_date = datetime.strptime(retrained_at, '%Y%m%d')
            days_old = (datetime.now() - retrain_date).days

            # Thresholds
            warning_threshold = 25  # Days
            critical_threshold = 40  # Days

            if days_old >= critical_threshold:
                status = 'critical'
                message = f'Models are {days_old} days old (>40 days)'
                should_retrain = True
            elif days_old >= warning_threshold:
                status = 'warning'
                message = f'Models are {days_old} days old (approaching 30-day target)'
                should_retrain = True
            else:
                status = 'ok'
                message = f'Models are {days_old} days old (within 25-day window)'
                should_retrain = False

            return {
                'status': status,
                'message': message,
                'days_old': days_old,
                'retrained_at': retrained_at,
                'should_retrain': should_retrain,
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to read metadata: {str(e)}',
                'days_old': None,
                'should_retrain': True,
            }

    def check_model_files(self) -> dict:
        """Check if all required model files exist."""

        expected_files = [
            'lgbm_1dte.pkl',
            'lgbm_2dte.pkl',
            'lgbm_5dte.pkl',
            'lgbm_10dte.pkl',
            'lgbm_20dte.pkl',
        ]

        missing_files = []
        for filename in expected_files:
            if not (self.prod_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            return {
                'status': 'error',
                'message': f'Missing model files: {", ".join(missing_files)}',
                'missing_files': missing_files,
            }

        return {
            'status': 'ok',
            'message': 'All required model files present',
            'missing_files': [],
        }

    def check_recent_performance(self) -> dict:
        """Check recent prediction performance if logs available."""

        # Look for most recent retraining results (ticker-specific)
        retrain_dirs = sorted(self.results_dir.glob(f'auto_retrain_{self.ticker}_*'), reverse=True)

        if not retrain_dirs:
            return {
                'status': 'unknown',
                'message': 'No recent performance data available',
                'rmse': None,
                'hit_rate': None,
            }

        latest_dir = retrain_dirs[0]
        summary_file = latest_dir / 'summary.csv'

        if not summary_file.exists():
            return {
                'status': 'unknown',
                'message': 'Performance summary not found',
                'rmse': None,
                'hit_rate': None,
            }

        try:
            df = pd.read_csv(summary_file)
            ensemble = df[df['method'] == 'ensemble_combined']

            if ensemble.empty:
                return {
                    'status': 'unknown',
                    'message': 'No ensemble performance data',
                    'rmse': None,
                    'hit_rate': None,
                }

            avg_rmse = ensemble['avg_midpoint_error'].mean()
            avg_hit_rate = ensemble['p99_hit_rate'].mean()

            # Performance thresholds
            if avg_rmse > 4.0 or avg_hit_rate < 95.0:
                status = 'warning'
                message = f'Performance degraded: RMSE={avg_rmse:.2f}%, Hit={avg_hit_rate:.1f}%'
            else:
                status = 'ok'
                message = f'Performance good: RMSE={avg_rmse:.2f}%, Hit={avg_hit_rate:.1f}%'

            return {
                'status': status,
                'message': message,
                'rmse': avg_rmse,
                'hit_rate': avg_hit_rate,
                'checked_date': latest_dir.name.replace('auto_retrain_', ''),
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to read performance data: {str(e)}',
                'rmse': None,
                'hit_rate': None,
            }

    def generate_report(self) -> dict:
        """Generate comprehensive health report."""

        age_check = self.check_model_age()
        files_check = self.check_model_files()
        perf_check = self.check_recent_performance()

        # Overall status
        statuses = [age_check['status'], files_check['status'], perf_check['status']]

        if 'error' in statuses or 'critical' in statuses:
            overall_status = 'critical'
            overall_message = '⚠️  CRITICAL: Immediate action required'
        elif 'warning' in statuses:
            overall_status = 'warning'
            overall_message = '⚠️  WARNING: Retraining recommended'
        else:
            overall_status = 'ok'
            overall_message = '✅ OK: Models healthy'

        return {
            'overall_status': overall_status,
            'overall_message': overall_message,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'age': age_check,
                'files': files_check,
                'performance': perf_check,
            },
            'recommendation': self._get_recommendation(age_check, files_check, perf_check),
        }

    def _get_recommendation(self, age_check, files_check, perf_check) -> str:
        """Generate recommendation based on checks."""

        if files_check['status'] == 'error':
            return 'URGENT: Retrain immediately - missing model files'

        if age_check.get('days_old', 0) > 40:
            return 'URGENT: Retrain immediately - models too old (>40 days)'

        if age_check.get('days_old', 0) > 25:
            return 'Retrain soon - models approaching 30-day threshold'

        # Handle None values in performance checks
        rmse = perf_check.get('rmse')
        if rmse is not None and rmse > 4.0:
            return 'Retrain recommended - RMSE degraded'

        hit_rate = perf_check.get('hit_rate')
        if hit_rate is not None and hit_rate < 95.0:
            return 'Retrain recommended - hit rate too low'

        return 'No action needed - models healthy'

    def print_report(self, report: dict):
        """Print formatted health report."""

        print("\n" + "=" * 80)
        print(f"MODEL HEALTH CHECK - {self.ticker}")
        print("=" * 80)
        print(f"Ticker: {self.ticker}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {report['overall_message']}")
        print()

        # Age check
        print("-" * 80)
        print("1. MODEL AGE")
        print("-" * 80)
        age = report['checks']['age']
        print(f"Status: {age['status'].upper()}")
        print(f"Message: {age['message']}")
        if age.get('days_old') is not None:
            print(f"Last retrained: {age.get('retrained_at', 'Unknown')}")
            print(f"Days old: {age['days_old']}")
        print()

        # Files check
        print("-" * 80)
        print("2. MODEL FILES")
        print("-" * 80)
        files = report['checks']['files']
        print(f"Status: {files['status'].upper()}")
        print(f"Message: {files['message']}")
        if files.get('missing_files'):
            print(f"Missing: {', '.join(files['missing_files'])}")
        print()

        # Performance check
        print("-" * 80)
        print("3. RECENT PERFORMANCE")
        print("-" * 80)
        perf = report['checks']['performance']
        print(f"Status: {perf['status'].upper()}")
        print(f"Message: {perf['message']}")
        if perf.get('rmse') is not None:
            print(f"RMSE: {perf['rmse']:.2f}% (threshold: 4.0%)")
            print(f"Hit Rate: {perf['hit_rate']:.1f}% (threshold: 95%)")
            print(f"Checked: {perf.get('checked_date', 'Unknown')}")
        print()

        # Recommendation
        print("=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(report['recommendation'])
        print()

        if report['overall_status'] in ['warning', 'critical']:
            print("Action: Run retraining script")
            print("  ./scripts/retrain_models_auto.sh")
        print("=" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(description='Monitor prediction model health')
    parser.add_argument('--ticker', type=str, default='NDX', choices=['NDX', 'SPX'],
                        help='Ticker symbol to check (default: NDX)')
    parser.add_argument('--alert-email', type=str, help='Email address for alerts')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    args = parser.parse_args()

    # Find project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Run health check
    monitor = ModelHealthMonitor(project_dir, ticker=args.ticker)
    report = monitor.generate_report()

    if args.json:
        # JSON output for programmatic use
        print(json.dumps(report, indent=2))
    else:
        # Human-readable output
        monitor.print_report(report)

    # Send email alert if requested and status is warning/critical
    if args.alert_email and report['overall_status'] in ['warning', 'critical']:
        try:
            import subprocess
            message = f"""
Model Health Alert

Status: {report['overall_status'].upper()}
Message: {report['overall_message']}

Recommendation: {report['recommendation']}

Details:
- Model age: {report['checks']['age'].get('days_old', 'Unknown')} days
- Files status: {report['checks']['files']['status']}
- Performance: RMSE={report['checks']['performance'].get('rmse', 'N/A')}%, Hit={report['checks']['performance'].get('hit_rate', 'N/A')}%

Action: Run ./scripts/retrain_models_auto.sh
            """

            subprocess.run(
                ['mail', '-s', f'Model Health Alert: {report["overall_status"].upper()}', args.alert_email],
                input=message.encode(),
                check=True,
            )
            print(f"✅ Alert sent to {args.alert_email}")
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")

    # Exit code based on status
    if report['overall_status'] == 'critical':
        sys.exit(2)
    elif report['overall_status'] == 'warning':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
