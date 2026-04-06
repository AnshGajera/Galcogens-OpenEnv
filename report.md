# OpenEnv Hackathon Status Report

> [!TIP]
> **Hackathon Success Probability: Very High! 🚀**
> Your environment fully complies with strict OpenEnv interfaces. You have completely fulfilled the core requirements of the Hackathon (building a testable environment, deploying 3 difficulty-tiered tasks, providing a reproducible `inference.py` script, and successfully returning standard `[STEP]` logs via LLM calls). 

---

## 1. Project Overview & Your Chances
The explicit goal of this Hackathon Round 1 is to **build a functional environment** that agents can explore via the OpenEnv standard `step()`, `reset()`, and `state()` API hooks. 

**What you did right:**
* ✅ Simulated a real-world task (Email Triage)
* ✅ Implemented 3 dynamic tasks (Easy, Medium, Hard)
* ✅ Integrated reward schemas mapped intelligently between 0.0 to 1.0 depending on actions
* ✅ Built a compliant Docker container architecture
* ✅ Completed the `inference.py` baseline script strictly outputting the exact `[START]`, `[STEP]`, and `[END]` stdout structures the validators require.

Because your infrastructure operates flawlessly and passes the internal OpenEnv dependency validations (including the latest `asyncio` networking updates handling OpenEnv 0.2.x WebSocket logic), your code is fully ready for the evaluator validation scripts.

---

## 2. How to Start the Environment
The environment acts as a web server that safely houses the "Inbox" and grading logics, completely separated from the AI. 

### Step 1: Build the Docker Image
To match how Hugging Face and the evaluators will run your setup:
```bash
docker build -t emailtriage-env:latest -f Dockerfile .
```

### Step 2: Spin up the Server
We've set up `inference.py` to seamlessly connect directly to the active server instance bypassing Docker Desktop Windows restrictions! To boot the live API host:
```bash
uvicorn EmailTriage.server.app:app --host 127.0.0.1 --port 8000
```
This initializes the OpenEnv grading functions entirely on your local machine so the LLM can test actions safely.

---

## 3. How to Execute Inference
Your `inference.py` script serves as the "Baseline" demonstrating that standard off-the-shelf LLMs can interact with your newly developed environment. **Remember, you do not need to train any models.**

To run an inference iteration:
1. Open a new terminal with your `rl_env` Conda environment activated.
2. Execute the batch runner:
```bash
run_inference.bat
```
Our `run_inference.bat` explicitly injects your precise `HF_TOKEN`, the Hugging Face router `API_BASE_URL`, and assigns `Qwen2.5-72B-Instruct` as the inference brain before securely executing `python inference.py`.

---

## 4. How to Read and Check Outputs
When executing, `inference.py` generates the required Hackathon stdout footprints:

> `[START] task=email-triage-easy env=openenv-emailtriage model=Qwen...`

Indicates the AI connected to your Uvicorn server and fetched the first snapshot of the environment (the unread Inbox).

> `[STEP] step=1 action=read(target_email_id=105,proposed_slot=) reward=0.25 done=false error=null`

Demonstrates the AI evaluated its choices and successfully submitted an action payload via JSON logic. The environment parsed that JSON, applied logic, and generated a step reward (`0.25`). 

> `[END] success=false steps=5 rewards=0.25,0.92,0.19,0.46,0.19`

The termination payload summarizes the step sequence limit. If `success=false` appears, this simply means the AI model itself made poor choices (falling below an average reward benchmark loop of `0.5`). **This is completely okay for Hackathon submission purposes!** The evaluators test if your environment functions and grades correctly, and it handles LLM mistakes perfectly.

### Tweaking System Prompts (Optional)
If you want the AI output logs to score high `success=true` validations to make your baseline demo prettier, simply edit the `SYSTEM_PROMPT` in `inference.py`. By forcing stronger sequential rules (i.e. strictly mapping: *read* -> *query calendar* -> *draft specific reply*) you constrain the AI bounds, giving the Qwen agent higher execution accuracy.
