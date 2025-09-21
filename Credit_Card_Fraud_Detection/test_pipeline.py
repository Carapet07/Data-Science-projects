#!/usr/bin/env python3
"""
Quick test script to validate the main pipeline before production run
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from data_preprocessing import DataLoadingVisualization, DataPreprocessing
        from model_building import DataPreparation, MLPBinary, ModelTrainer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directories():
    """Test if all required directories exist or can be created"""
    print("Testing directory structure...")
    project_dir = Path(__file__).parent
    required_dirs = ['data_cache', 'saved_models', 'logs']
    
    for dir_name in required_dirs:
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        if dir_path.exists():
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Failed to create directory {dir_name}")
            return False
    return True

def main():
    """Run all tests"""
    print("="*50)
    print("PIPELINE VALIDATION TEST")
    print("="*50)
    
    tests = [
        ("Module imports", test_imports),
        ("Directory structure", test_directories),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Pipeline is ready to run.")
        print("\nTo run the full training pipeline:")
        print("cd /Users/romacarapetean/Desktop/Projects/Credit_Card_Fraud_Detection")
        print("source .venv/bin/activate")
        print("python src/main.py")
    else:
        print("❌ Some tests failed. Please fix issues before running pipeline.")
    print("="*50)

if __name__ == "__main__":
    main()
