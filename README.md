# Genesis

Genesis is a student-built Python project that combines a small machine-learning workflow with an LLM-guided tool interface and a Streamlit UI.

The current version can:

- Load a CSV from a direct URL or a Hugging Face dataset source.

- Train a scikit-learn `RandomForestClassifier` on a selected target column.

- Apply basic categorical encoding with pandas and label encoding for text targets.

- Save the trained model locally as a Joblib file.

- Log the training accuracy to Weights & Biases when configured.

- Generate a confusion matrix and feature-importance plot.

- Expose the workflow through a Streamlit interface using Groq tool calling.

- Provide a separate text/voice-oriented command loop in `agent.py`.

This repository is an active learning project. It is not presented as a production-ready autonomous agent. The next engineering improvements are reproducible setup, tests, preprocessing consistency, CI, and a measured end-to-end example.

## Project status

| Area | Current status |
| --- | --- |
| Dataset ingestion | Implemented for direct CSV URLs and Hugging Face dataset IDs |
| Model training | Implemented with a Random Forest classifier |
| Evaluation | Accuracy, confusion matrix, and feature-importance output are implemented |
| Model persistence | Local Joblib artifact is implemented |
| Experiment tracking | Weights & Biases logging is implemented when configured |
| User interface | Streamlit interface is present |
| Tests | Not yet added |
| Continuous integration | Not yet added |
| Public deployment | Not yet available |
| Reproducible benchmark | Not yet published |

## Repository structure

```
.
├── agent.py                 # Dataset, training, evaluation, and command-loop logic
├── app.py                   # Streamlit interface and LLM tool-calling loop
├── predict.py               # Local prediction helper
├── tracker.py               # Experiment-tracking helper
├── data/                    # Example CSV datasets
├── models/                  # Locally generated model artifacts
├── plots/                   # Locally generated evaluation plots
├── command.wav              # Local audio sample
├── genesis interface.css    # Streamlit UI styling
├── requirements.txt         # Pinned Python dependencies currently used by the project
└── README.md
```

## Current workflow

```
CSV URL or Hugging Face dataset
            │
            ▼
      Dataset ingestion
            │
            ▼
  Basic preprocessing and encoding
            │
            ▼
  Random Forest train/test split
            │
            ├── Local Joblib model
            ├── Accuracy logged to W&B
            └── Confusion matrix and feature-importance plots
```

The LLM interface can select the ingestion, training, and evaluation tools from natural-language commands. The underlying model workflow remains a conventional scikit-learn pipeline.

## Prerequisites

- Python 3.10 or later is recommended. The project currently needs an environment check because the dependency file and source imports are still being aligned.

- A Groq API key is needed for the LLM-powered interface and voice transcription path.

- A Weights & Biases API key is needed if experiment logging is enabled.

- A microphone is required only for the optional voice-command path.

Never commit API keys to the repository. Use environment variables or a local `.env` file that is excluded from Git.

## Local setup

The setup commands below describe the intended workflow. The dependency list is being cleaned and tested as part of the next project iteration.

```bash
git clone https://github.com/FirstAnsh/genesis.git
cd genesis
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file only after verifying the required packages and variables for the path you want to run:

```
GROQ_API_KEY=your_groq_key
WANDB_API_KEY=your_wandb_key
```

The current Streamlit entry point is:

```bash
streamlit run app.py
```

The command-loop entry point is:

```bash
python agent.py
```

If either entry point fails in a fresh environment, please open an issue with the Python version, operating system, command used, and full error message. Do not include API keys in issue reports.

## Example tools

The current tool interface exposes three operations:

1. `fetch_dataset` — downloads a dataset and saves it under `data/`.

1. `train_random_forest` — trains a Random Forest using a CSV path and target column.

1. `evaluate_model` — loads a saved model and generates evaluation plots.

Example commands for the interface will be added after the project has a fixed, reproducible sample dataset and a verified clean-environment run.

## Limitations of the current version

- Preprocessing is intentionally simple and is not yet packaged as a single reusable fitted pipeline.

- The evaluator recreates preprocessing independently, so feature alignment and categorical handling need stronger safeguards.

- The primary reported metric is accuracy; additional metrics and a documented baseline are still needed.

- There are no automated tests or CI checks yet.

- The UI contains prototype dashboard text and should not be interpreted as evidence of completed tasks or production monitoring.

- External APIs can fail, incur cost, or change behaviour. The project currently needs clearer timeouts, validation, and error handling around those integrations.

- The project has no public hosted demo at this time.

## Roadmap

- [ ] Clean and verify the dependency list in a fresh virtual environment.

- [ ] Add a fixed public example dataset and one reproducible benchmark run.

- [ ] Add tests for ingestion, missing columns, preprocessing, training, evaluation, and error paths.

- [ ] Persist preprocessing together with the model to prevent feature mismatch.

- [ ] Add metrics beyond accuracy and document the train/test split.

- [ ] Add GitHub Actions for tests and basic quality checks.

- [ ] Add screenshots or a short demo after the workflow is verified.

- [ ] Deploy a small demo only after the local workflow is stable.

## Author

Ansh Yewale — BTech CSE (AI-ML) student, Vidharbha Institute of Technology, Nagpur.

- GitHub: [https://github.com/FirstAnsh](https://github.com/FirstAnsh)

- LinkedIn: [https://www.linkedin.com/in/ansh-yewale-3293653a3/](https://www.linkedin.com/in/ansh-yewale-3293653a3/)
