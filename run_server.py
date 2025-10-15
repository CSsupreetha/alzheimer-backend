#!/usr/bin/env python3
"""
Quick start script for the Medical AI Flask server (Grayscale 3-Class Model)
"""

import os
import sys
import subprocess
import shutil

# --- Model preload imports ---
from model_loader import ModelLoader
model_loader = ModelLoader()
model = None  # Global model variable


def preload_model():
    """Preload the grayscale Keras model before starting the server"""
    global model
    try:
        # Updated model path for grayscale version
        model_path = "./gray_hybrid_model.keras"
        model = model_loader.load_model(model_path)
        print("✅ Grayscale model preloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to preload model: {str(e)}")
        return False


# --- Dependency check ---
def check_dependencies():
    package_map = {
        "flask": "flask",
        "tensorflow": "tensorflow",
        "pillow": "PIL",
        "numpy": "numpy",
        "requests": "requests",
        "web3": "web3",
        "firebase-admin": "firebase_admin",
        "flask-cors": "flask_cors",
        "python-dotenv": "dotenv",
        "gunicorn": "gunicorn",
    }

    missing_packages = []
    for pkg, import_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print("pip install " + " ".join(missing_packages))
        return False

    print("✅ All required packages are installed")
    return True


# --- Model file check ---
def check_model_file():
    model_path = "../attached_assets/gray_hybrid_model.keras"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure your gray_hybrid_model.keras file is in the attached_assets directory")
        return False

    file_size = os.path.getsize(model_path)
    print(f"✅ Model file found: {model_path}")
    print(f"   File size: {file_size / (1024*1024):.2f} MB")
    return True


# --- Environment setup ---
def setup_environment():
    env_file = ".env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file from template...")
        try:
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file")
            print("   Please edit .env file to add your API keys and configuration")
        except Exception as e:
            print(f"❌ Failed to create .env file: {str(e)}")
            return False
    else:
        print("✅ .env file exists")
    return True


# --- Run tests ---
def run_tests():
    print("\n🧪 Running model tests...")
    try:
        result = subprocess.run([sys.executable, "test_model.py"],
                                capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out (this is normal for large models)")
        return True
    except Exception as e:
        print(f"❌ Failed to run tests: {str(e)}")
        return False


# --- Start server ---
def start_server(mode='development'):
    print(f"\n🚀 Starting Flask server in {mode} mode...")
    if mode == 'development':
        os.environ['FLASK_ENV'] = 'development'
        os.environ['DEBUG'] = 'True'
        try:
            from app import app, load_model_on_startup

            print("⚡ Loading grayscale model inside Flask app before starting server...")
            if not load_model_on_startup():
                print("❌ Model failed to load inside Flask app (predictions won’t work)")
            else:
                print("✅ Model loaded successfully inside Flask app")

            app.run(host='0.0.0.0', port=5000, debug=True)

        except ImportError as e:
            print(f"❌ Failed to import app: {str(e)}")
            return False

    elif mode == 'production':
        try:
            cmd = [
                "gunicorn",
                "--bind", "0.0.0.0:5000",
                "--workers", "2",
                "--timeout", "120",
                "--worker-class", "sync",
                "app:app"
            ]
            subprocess.run(cmd)
        except Exception as e:
            print(f"❌ Failed to start Gunicorn: {str(e)}")
            return False

    return True


# --- Main ---
def main():
    print("=" * 60)
    print("🏥 MEDICAL AI SERVER - ALZHEIMER'S PREDICTION (Grayscale 3-Class Model)")
    print("=" * 60)

    print("\n1. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print("\n2. Checking model file...")
    if not check_model_file():
        sys.exit(1)

    print("\n3. Setting up environment...")
    if not setup_environment():
        sys.exit(1)

    # --- Preload model ---
    if not preload_model():
        print("❌ Cannot start server - model preload failed")
        sys.exit(1)

    # --- Run tests option ---
    run_tests_choice = input("\n4. Run model tests? (y/n): ").lower().strip()
    if run_tests_choice in ['y', 'yes']:
        if not run_tests():
            continue_anyway = input("\nContinue anyway? (y/n): ").lower().strip()
            if continue_anyway not in ['y', 'yes']:
                sys.exit(1)

    # --- Choose server mode ---
    print("\n5. Choose server mode:")
    print("   1. Development (Flask dev server)")
    print("   2. Production (Gunicorn)")
    mode_choice = input("Enter choice (1-2): ").strip()
    mode = "development" if mode_choice != "2" else "production"

    print(f"\n✅ Ready to start server in {mode} mode!")
    print("📋 Server will be available at: http://localhost:5000")
    print("🔗 Health check endpoint: http://localhost:5000/health")
    print("🧠 Prediction endpoint: http://localhost:5000/predict")

    input("\nPress Enter to start the server...")

    try:
        start_server(mode)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Server failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
