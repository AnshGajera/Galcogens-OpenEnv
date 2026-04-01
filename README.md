# Email Triage & Workflow Orchestration Agent

A reinforcement learning agent built with [OpenEnv](https://github.com/open-env/openenv) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) that learns to intelligently manage email workflows. The agent handles tasks ranging from spam filtering to drafting meeting invitations and resolving ambiguous client requests.

This project was developed for the **OpenEnv Hackathon** (Meta PyTorch track) and demonstrates a production‑ready agent that can be deployed via Hugging Face Hub with a simple inference script.

---

##  Project Overview

The environment simulates a realistic email inbox where the agent receives observations (email metadata, conversation history, calendar availability) and must take actions such as:
- **Archive low‑priority emails** (spam, automated notifications)
- **Draft replies** for meeting requests using a calendar tool
- **Resolve ambiguous messages** by retrieving context from past conversations

The agent learns through reward signals that evaluate correctness, professionalism, and adherence to user preferences. All interactions are defined using **Pydantic models** for strict validation, ensuring the environment meets the hackathon’s technical requirements.

---

## Features

- **Multi‑step, stateful environment** – Maintains conversation history and user metadata.
- **Deterministic grading** – Reward functions based on exact action matching (e.g., SQL flag updates, Pydantic model validation).
- **Partial progress rewards** – Encourages intermediate successes (e.g., identifying the correct project name).
- **Hugging Face integration** – Upload trained models to the Hub and run inference with a standalone script.
- **Lightweight & efficient** – Fits within 8 GB RAM / 2 vCPU limits; inference runs in < 20 minutes.

---

##  Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AnshGajera/Galcogens-OpenEnv
   cd email-triage-rl
   ```

2. **Create and activate the Conda environment**
   ```bash
   conda create -n rl_env python=3.10 -y
   conda activate rl_env
   ```
   *Use Command Prompt (cmd) – not PowerShell – to avoid activation issues.*

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install gymnasium stable-baselines3 huggingface_hub matplotlib pygame
   pip install -r requirements.txt
   ```

4. **Install IPython kernel for Jupyter (optional)**
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name rl_env --display-name "RL Environment"
   ```

---

##  Usage

### Train the agent
```bash
python train.py
```
The training script will:
- Create the OpenEnv environment.
- Initialize a PPO agent (from Stable‑Baselines3).
- Train for a specified number of timesteps.
- Save the final model to `models/` and upload it to the Hugging Face Hub (if credentials are set).

### Run inference
```bash
python inference.py --model_path "your-hf-username/email-triage-model" --email "New client email..."
```
The `inference.py` script loads the model from the Hub, processes the input email, and outputs the chosen action (e.g., archived message, drafted reply). It also logs the reward achieved on that step.

---

##  Environment Details

The environment is implemented using OpenEnv and follows the Gymnasium interface. Key components:

- **Observations**: Pydantic model with fields like `sender`, `subject`, `body`, `conversation_history`, `calendar_slots`, `user_metadata`.
- **Actions**: Pydantic model with fields such as `action_type` (archive, reply, escalate), `reply_text`, `calendar_event_id`.
- **Reward function**:
  - `+1.0` for correctly archiving a low‑priority email.
  - `+0.5` for correctly identifying a project name in an ambiguous email.
  - `+2.0` for drafting a professional meeting request with a valid time slot.
  - `-0.5` for wrong actions.
  - Additional penalties for rule violations (e.g., forgetting to check calendar availability).

The environment includes three difficulty levels (Easy, Medium, Hard) to evaluate progressive learning.

---

##  Deployment

The trained model can be deployed to the Hugging Face Hub with:
```bash
python upload_to_hub.py --model_path models/ppo_email_triage.zip --repo_name email-triage-model
```

For inference in production, use `inference.py` (provided in the repo) which:
- Loads the model from Hugging Face.
- Parses incoming email data (JSON or plain text).
- Returns the action as a structured Pydantic object.
- Runs in < 20 minutes per request (typically a few seconds).

---

##  Project Structure

```
email-triage-rl/
├── envs/                     # OpenEnv environment definition
│   ├── email_env.py
│   └── models.py             # Pydantic schemas for obs/actions
├── train.py                  # Training script
├── inference.py              # Inference script
├── upload_to_hub.py          # Upload model to Hugging Face
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # Saved models (ignored by git)
```
OR 

```
├── 📝 models.py          ← Type-safe contracts
│                           (Action, Observation, State)
│
├── 📱 client.py          ← What YOU import
│                           (HTTPEnvClient implementation)
│
└── 🖥️  server/
    ├── environment.py    ← Game/simulation logic
    ├── app.py            ← FastAPI server
    └── Dockerfile        ← Container definition
```


---

##  Contributing

Feel free to open issues or pull requests for improvements. For the hackathon, ensure all environment actions use Pydantic validation and that the inference script meets the 20‑minute runtime requirement.

---

##  License

MIT

---

##  Acknowledgements

- [OpenEnv](https://github.com/open-env/openenv) – Environment framework
- [Stable‑Baselines3](https://stable-baselines3.readthedocs.io/) – RL algorithms
- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) – Model hosting
- Meta PyTorch Hackathon for the inspiration and problem statements