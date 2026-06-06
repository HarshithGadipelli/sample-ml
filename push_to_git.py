import subprocess
import os

def run_command(cmd):
    print(f"\n> {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"Failed to run command: {e}")

def main():
    print("--- Starting Git Push ---")
    
    # 1. Initialize git
    run_command(["git", "init"])
    
    # 2. Add all files
    run_command(["git", "add", "."])
    
    # 3. Commit
    run_command(["git", "commit", "-m", "Major update: Scaled to 500 Telangana colleges, advanced search, and new glassmorphism UI"])
    
    # 4. Set main branch
    run_command(["git", "branch", "-M", "main"])
    
    # 5. Add remote (ignore error if it already exists)
    run_command(["git", "remote", "add", "origin", "https://github.com/HarshithGadipelli/ML-CCC.git"])
    
    # 6. Push to GitHub
    print("\nPushing to GitHub (this may take a moment)...")
    run_command(["git", "push", "-u", "origin", "main"])
    
    print("\n--- Done! ---")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
