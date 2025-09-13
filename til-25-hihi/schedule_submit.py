import time
import subprocess

def run_command():
    try:
        subprocess.run(['til', 'submit', 'hihi-rl:finals'], check=True)
        print("Command ran successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")

if __name__ == "__main__":
    while True:
        run_command()
        print("Waiting 8 minutes...")
        time.sleep(8 * 60)
