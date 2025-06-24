"""
Script to run all tests for the Legal Contract Analyzer.
"""

import unittest
import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def discover_tests() -> unittest.TestLoader:
    """Discover all test cases in the tests directory."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent.parent / "tests"
    return loader.discover(str(start_dir), pattern="test_*.py")

def run_tests(
    verbosity: int = 2,
    failfast: bool = False,
    buffer: bool = True
) -> unittest.TestResult:
    """Run all discovered tests."""
    # Create test suite
    suite = discover_tests()
    
    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        buffer=buffer
    )
    
    # Run tests
    logger.info("Starting test execution...")
    result = runner.run(suite)
    logger.info("Test execution completed.")
    
    return result

def generate_report(result: unittest.TestResult) -> Dict:
    """Generate a detailed test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        "failures": [
            {
                "test": str(failure[0]),
                "message": str(failure[1])
            }
            for failure in result.failures
        ],
        "errors": [
            {
                "test": str(error[0]),
                "message": str(error[1])
            }
            for error in result.errors
        ],
        "skipped": [
            {
                "test": str(skip[0]),
                "message": str(skip[1])
            }
            for skip in result.skipped
        ]
    }
    
    return report

def save_report(report: Dict, output_dir: Optional[Path] = None) -> None:
    """Save test report to file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "reports"
    
    # Create reports directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"test_report_{timestamp}.json"
    
    # Save report
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to: {report_file}")

def main():
    """Main function to run tests and generate report."""
    try:
        # Run tests
        result = run_tests()
        
        # Generate and save report
        report = generate_report(result)
        save_report(report)
        
        # Print summary
        logger.info("\nTest Summary:")
        logger.info(f"Total Tests: {report['total_tests']}")
        logger.info(f"Failures: {report['failures']}")
        logger.info(f"Errors: {report['errors']}")
        logger.info(f"Skipped: {report['skipped']}")
        logger.info(f"Success Rate: {report['success_rate']:.2%}")
        
        # Exit with appropriate status code
        sys.exit(1 if (report['failures'] > 0 or report['errors'] > 0) else 0)
    
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 