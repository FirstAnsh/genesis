import os
import json
import wandb
import pandas as pd
import numpy as np
import sounddevice as sd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
load_dotenv()

# --- Tool 1: Multi-Source Data Ingestion ---
def fetch_dataset(dataset_name: str):
    """Downloads a dataset from Hugging Face Hub or a direct CSV URL."""
    try:
        print(f"[SYSTEM EXECUTION] Fetching dataset '{dataset_name}'...")
        
        # If the AI is given a web link, bypass Hugging Face and download directly
        if dataset_name.startswith("http"):
            df = pd.read_csv(dataset_name)
        else:
            # Otherwise, use standard Hugging Face Parquet loading
            ds = load_dataset(dataset_name, split="train")
            df = ds.to_pandas()
        
        os.makedirs("data", exist_ok=True)
        safe_name = dataset_name.split('/')[-1].replace('.csv', '')
        local_path = f"data/{safe_name}_data.csv"
        df.to_csv(local_path, index=False)
        
        return json.dumps({
            "dataset_name": dataset_name,
            "num_rows": len(df),
            "columns": list(df.columns),
            "csv_path": local_path,
            "status": "success"
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# --- Tool 2: Dynamic Model Training ---
def train_random_forest(csv_path: str, target_column: str):
    """Trains a Random Forest on a specific CSV file and target column, logging to W&B."""
    try:
        print(f"[SYSTEM EXECUTION] Training model on {csv_path} predicting '{target_column}'...")
        wandb.init(project="genesis-agent", name=f"rf-{target_column}")
        
        df = pd.read_csv(csv_path)
        
        cols_to_drop = [col for col in df.columns if col.lower() in ['id', 'unnamed: 0'] or col == target_column]
        X = df.drop(columns=cols_to_drop)
        y = df[target_column]
        # -------------------------------------------------------------------
        
        # Encode categorical targets (e.g., text labels into numbers)
        if y.dtype == 'object':
             y = df[target_column]
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
        X = pd.get_dummies(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = f"models/rf_{target_column}.joblib"
        joblib.dump(clf, model_path)
        
        preds = clf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        wandb.log({"accuracy": accuracy, "target": target_column})
        wandb.finish()
        
        return json.dumps({
            "status": "Training complete",
            "accuracy": accuracy,
            "model_path": model_path
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# --- Tool 3: Voice Command Transcription ---
def get_voice_command(client):
    """Records audio from the microphone and transcribes it using Groq Whisper."""
    fs = 44100
    duration = 6
    
    print(f"\n🎤 Recording for {duration} seconds. Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    
    wav_path = "command.wav"
    write(wav_path, fs, recording)
    print("✅ Audio captured. Transcribing via Groq Whisper...")
    
    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    
    print(f"[WHISPER TRANSCRIPT]: {transcription.strip()}")
    return transcription.strip()
# --- Tool 4: Model Visual Evaluation ---
def evaluate_model(model_path: str, csv_path: str, target_column: str):
    """Evaluates a trained model and saves confusion matrix and feature importance plots."""
    try:
        print(f"[SYSTEM EXECUTION] Generating evaluation plots for '{target_column}'...")
        
        model = joblib.load(model_path)
        df = pd.read_csv(csv_path)
        
        # Apply the exact same preprocessing used in training
        cols_to_drop = [col for col in df.columns if col.lower() in ['id', 'unnamed: 0'] or col == target_column]
        X = df.drop(columns=cols_to_drop)
        y = df[target_column]
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preds = model.predict(X_test)
        os.makedirs("plots", exist_ok=True)
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = f"plots/cm_{target_column}.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Plot 2: Feature Importance Plot
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10] # Top 10 features
        
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [model.feature_names_in_[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        fi_path = f"plots/fi_{target_column}.png"
        plt.savefig(fi_path)
        plt.close()
        
        return json.dumps({
            "status": "Evaluation plots generated",
            "confusion_matrix": cm_path,
            "feature_importance": fi_path
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# --- Agent Orchestration Loop ---
# --- Agent Orchestration Loop ---
def boot_agent_brain():
    print("Initializing Genesis Persistent Text Interface...")
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    {
            "type": "function",
            "function": {
                "name": "evaluate_model",
                "description": "Evaluates a trained model and generates confusion matrix and feature importance PNG plots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_path": {"type": "string", "description": "Local path to the .joblib model file"},
                        "csv_path": {"type": "string", "description": "Local path to the CSV dataset"},
                        "target_column": {"type": "string", "description": "The exact name of the target column"}
                    },
                    "required": ["model_path", "csv_path", "target_column"]
                }
            }
        }
    
    tools = [
        
        {
            "type": "function",
            "function": {
                "name": "fetch_dataset",
                "description": "Fetch a dataset from Hugging Face Hub and save it locally as CSV.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {"type": "string", "description": "Hugging Face dataset ID or a direct CSV URL."}
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "train_random_forest",
                "description": "Trains a Random Forest classifier on a local CSV.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "csv_path": {"type": "string", "description": "The local file path to the CSV data"},
                        "target_column": {"type": "string", "description": "The exact name of the column to predict"}
                    },
                    "required": ["csv_path", "target_column"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_model",
                "description": "Evaluates a trained model and saves confusion matrix and feature importance plots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_path": {"type": "string", "description": "The local file path to the trained model"},
                        "csv_path": {"type": "string", "description": "The local file path to the CSV data"},
                        "target_column": {"type": "string", "description": "The exact name of the column to predict"}
                    },
                    "required": ["model_path", "csv_path", "target_column"]
                }
            }
        }
    ]
    
    tool_map = {
        "fetch_dataset": fetch_dataset,
        "train_random_forest": train_random_forest,
        "evaluate_model": evaluate_model
    }
    # ---------------------------

    system_prompt = (
        "You are Genesis, an autonomous ML engineering agent. "
        # ...
    )
    
    system_prompt = (
        "You are Genesis, an autonomous ML engineering agent. "
        "You can fetch datasets from Hugging Face and train Random Forest models on local CSVs. "
        "Wait for the user's command, execute the necessary tools, and report back clearly. "
        "CRITICAL INSTRUCTION: If a tool returns an error, you MUST explain the error to the user in natural language. Do not output an empty response."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    print("\n========================================")
    print("🧠 Genesis is online. Type 'exit' to quit.")
    print("========================================\n")
    
   # Outer loop: Waits for your commands (Text or Voice)
    while True:
        user_input = input("You (Type a command, or press Enter to use Voice): ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Genesis shutting down...")
            break
            
        # If you just hit Enter, activate the microphone
        if user_input.strip() == "":
            user_input = get_voice_command(client)
            if not user_input:
                continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Inner loop: Genesis thinks and uses tools
        while True:
            response = client.chat.completions.create(
                # Use Groq's current flagship 120B model:
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )
            
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)
            
            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    t_name = tool_call.function.name
                    t_args = json.loads(tool_call.function.arguments or "{}")
                    print(f"\n[ACTION INTENT] Genesis calling: {t_name}({t_args})")
                    
                    if t_name in tool_map:
                        tool_output = tool_map[t_name](**t_args)
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})
            else:
                response_text = assistant_msg.content if assistant_msg.content else "[System Error: Genesis returned an empty response.]"
                print(f"\nGenesis: {response_text.strip()}\n")
                break

if __name__ == "__main__":
    boot_agent_brain()