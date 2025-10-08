import os
import sys


def setup_environment():
    """Set up Python path and environment variables"""

    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Set environment variables
    os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

    print(f"Project root added to path: {project_root}")


if __name__ == "__main__":
    setup_environment()