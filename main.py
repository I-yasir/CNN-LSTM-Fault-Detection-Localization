import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n>> Executing: {script_path}")
    
    # Copy the current environment variables
    env = os.environ.copy()
    # Add the project root to PYTHONPATH so scripts can find the 'src' module
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")
    
    # Pass the updated environment to the subprocess
    result = subprocess.run([sys.executable, script_path], env=env)
    
    if result.returncode != 0:
        print(f"Error executing {script_path}. Halting pipeline.")
        sys.exit(1)

def main():
    print("========================================")
    print(" Pipeline: Fault Detection & Localization")
    print("========================================\n")
    
    scripts = [
        "scripts/01_split.py",
        "scripts/02_normalize.py",
        "scripts/03_train.py",
        "scripts/04_evaluate.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n========================================")
    print(" Pipeline finished successfully.")
    print("========================================")

if __name__ == "__main__":
    main()