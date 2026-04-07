Project Context
The Organizers: This hackathon is a collaborative effort involving Scaler School of Technology, Meta PyTorch OpenEnv, and Hugging Face
.
The Core Problem: Currently, the AI industry lacks standardized reinforcement learning (RL) environments, making them difficult to scale in post-training runs
. The OpenEnv framework aims to solve this by providing a unified, interoperable API contract
.
The Goal of RL: Reinforcement learning improves language models by updating weights via backpropagation using reward signals, rather than relying on massive, inefficient context windows (in-context learning)
.
Your Role: You are building the "testing ground" (the environment) that a frontier AI lab could theoretically plug into a massive training pipeline to teach an agent how to complete a specific domain task
.
What You Need to Build
A Real-World RL Environment: You must build a simulated environment based on a valuable, real-world task (e.g., healthcare APIs, flight booking, or proactive email triage)
.
The Core Logic (models.py): You must define your environment's logic by extending base OpenEnv types using strict Pydantic objects for the Action, Observation, and State
.
The Grader (Reward System): Your environment must include a customized reward mechanism that evaluates the agent's actions and returns a continuous, diverse reward score strictly between 0 and 1
.
The Inference Script (inference.py): You must provide a mandatory test script that runs the interaction loop, proving that an LLM can successfully navigate your environment and earn a reward
.
Strict Rules and Constraints
No Toy Games: Your submission must simulate a real-world task. Simple games like Wordle, Connect 4, or number guessers are strictly prohibited and will not be evaluated
.
No Paid OpenAI Keys: You must not use or pay for an OpenAI API key. You must use a free Hugging Face token and the built-in Hugging Face router in your inference script, which auto-selects and manages the model calls
.
Originality is Mandatory: Plagiarism is strictly checked. You cannot copy existing environments from the open-source hub; your task and logic must be novel
.
Dockerfile Placement: You must manually move the generated Dockerfile out of the server folder and place it directly into your outermost root folder
.
Inference Script is Mandatory: If your inference.py script is missing or fails to run, the judges cannot evaluate your environment, rendering your submission invalid
.
Deadline: The final deadline for submissions is April 8th
. You can submit multiple times; the evaluators will only grade your most recent working submission
.
Evaluation & Success Criteria
Real-World Utility: Does the environment solve a documented, practical pain point that researchers would actually want to train an AI on?
.
Quality of the Grader: Does the grader provide a rich, diverse learning landscape (e.g., offering partial rewards for progress) rather than just giving the same static score every time?
.
Long-Running Tasks: High-quality environments force the model to adapt to dynamically changing states over multiple turns, rather than just solving single-turn, static problems
.
Tooling and Development Workflow
OpenEnv CLI: You will use the openenv-core library to handle initialization (openenv init), validation (openenv validate), and deployment (openenv push)
.
Hugging Face Spaces: Your final product will be deployed as a Hugging Face Space. This acts as your API, UI (via Gradio), version control, and Docker registry all in one
.
AI Assistance is Encouraged: You are fully allowed and encouraged to use AI coding assistants (like Claude, Cursor, or ChatGPT) to help you build the environment and update the mandatory inference script, provided the core idea is your own
