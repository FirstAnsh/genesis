import wandb
import os
from dotenv import load_dotenv

# Load API keys securely from the .env file
load_dotenv()

def initialize_genesis():
    print("Booting Genesis architecture...")
    
    # Log in to W&B using the key from .env
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    # Initialize a W&B run to track our future agent's metrics
    run = wandb.init(
        project="genesis-agent",
        name="baseline-init",
        notes="Initial environment and telemetry test for the autonomous ML agent."
    )
    
    print(f"Tracking active. Run ID: {run.id}")
    
    # Close the run cleanly
    wandb.finish()
    print("Run completed and synced to the cloud.")

if __name__ == "__main__":
    initialize_genesis()