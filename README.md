# VibeCoding-homework-01 – "What Next?" Activity Recommender

> **Czech / Česky:** Python skript, který pomocí OpenAI Function Calling (Tool Use) doporučuje aktivity na základě polohy, věku a denní doby uživatele.
>
> **English:** A Python application that uses OpenAI's Function Calling (Tool Use) to recommend activities based on the user's location, age, and current time.

---

## Table of Contents / Obsah

1. [What does this project do?](#1-what-does-this-project-do)
2. [How does Tool Use / Function Calling work?](#2-how-does-tool-use--function-calling-work)
3. [Project structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Setup – step by step (for beginners)](#5-setup--step-by-step-for-beginners)
   - [Option A – Remote OpenAI API](#option-a--remote-openai-api-requires-an-api-key)
   - [Option B – Local AI with Ollama (free, no API key)](#option-b--local-ai-with-ollama-free-no-api-key)
   - [Option C – Local AI with LM Studio (free, no API key)](#option-c--local-ai-with-lm-studio-free-no-api-key)
6. [Running the application](#6-running-the-application)
7. [Example conversation](#7-example-conversation)
8. [Running with GitHub Copilot Agent / Remote Agent](#8-running-with-github-copilot-agent--remote-agent)
9. [Troubleshooting](#9-troubleshooting)
10. [How to get an OpenAI API key](#10-how-to-get-an-openai-api-key-optional)

---

## 1. What does this project do?

You chat with an AI assistant and it recommends activities (cinema, gym, yoga, hiking, etc.) that are:

- **Age-appropriate** for you.
- **Available right now** (outdoor activities are not offered late at night).
- **Suited to the current season** (e.g. swimming in summer, ice skating in winter).

The assistant speaks in **Czech or English** (or any language you use).

---

## 2. How does Tool Use / Function Calling work?

Traditional LLMs can only generate text. **Tool Use** (also called *Function Calling*) lets the model call a Python function during a conversation to fetch real data.

```
User says:  "I am 25 years old, I am in Prague, Wenceslas Square"
                          │
                          ▼
          ┌───────────────────────────────┐
          │  OpenAI GPT-4o-mini           │
          │  decides to call a tool:      │
          │  get_recommendations(         │
          │    location="Prague, …",      │
          │    age=25,                    │
          │    current_time="14:30",      │
          │    season="spring"            │
          │  )                            │
          └───────────┬───────────────────┘
                      │ [TOOL CALL] logged to console
                      ▼
          ┌───────────────────────────────┐
          │  Python function runs and     │
          │  returns a list of activities │
          └───────────┬───────────────────┘
                      │ [TOOL RESULT] logged to console
                      ▼
          ┌───────────────────────────────┐
          │  GPT-4o-mini reads the result │
          │  and writes a friendly reply  │
          └───────────┬───────────────────┘
                      │ [ASSISTANT RESPONSE]
                      ▼
        "Here are some ideas for you: …"
```

The console will print **`[TOOL CALL]`** whenever the model invokes a function and **`[TOOL RESULT]`** when Python returns the data, so you can follow every step.

---

## 3. Project structure

```
VibeCoding-homework-01/
├── main.py           ← All application code (LLM logic + tool definition)
├── requirements.txt  ← Python dependencies
├── .env.example      ← Template for your API key (rename to .env)
├── .env              ← Your actual API key – NOT committed to git (gitignored)
├── .gitignore        ← Files/folders excluded from git
├── LICENSE
└── README.md         ← This file
```

---

## 4. Prerequisites

The application supports **two modes**. Choose the one that fits you:

| Mode | What you need | Cost |
|---|---|---|
| **Remote OpenAI API** | Python 3.10+, OpenAI API key | Paid (free trial credits available) |
| **Local AI – Ollama** | Python 3.10+, Ollama installed | Free |
| **Local AI – LM Studio** | Python 3.10+, LM Studio installed | Free |

> **Windows users:** During Python installation tick **"Add Python to PATH"**.

---

## 5. Setup – step by step (for beginners)

### Step 1 – Download the project

**Option A – with Git** (recommended):
```bash
git clone https://github.com/HRADEPA1/VibeCoding-homework-01.git
cd VibeCoding-homework-01
```

**Option B – without Git:**
1. Go to https://github.com/HRADEPA1/VibeCoding-homework-01
2. Click the green **Code** button → **Download ZIP**.
3. Unzip the file and open a terminal in the resulting folder.

### Step 2 – Create a virtual environment (optional but recommended)

A virtual environment keeps the project's dependencies isolated from the rest of your system.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You will see `(.venv)` at the start of your prompt when it is active.

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **openai** – the official OpenAI Python SDK (also works with local servers).
- **python-dotenv** – reads the `.env` file so credentials stay out of the code.

### Step 4 – Configure your AI backend

Copy the example file first:
```bash
# macOS / Linux
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Then open `.env` and follow **one** of the three options below.

> **Never share your `.env` file or commit it to GitHub.** It is already listed in `.gitignore`.

---

#### Option A – Remote OpenAI API (requires an API key)

Edit `.env` so it contains:
```
OPENAI_API_KEY=sk-...your-key-here...
# MODEL_NAME=gpt-4o-mini   ← optional, this is the default
```

The app will call OpenAI's servers. You need a paid account or free-trial credits.
See [section 10](#10-how-to-get-an-openai-api-key-optional) for how to get a key.

---

#### Option B – Local AI with Ollama (free, no API key)

[Ollama](https://ollama.com) runs open-source models on your own computer.
No API key, no internet needed after the model is downloaded.

1. **Install Ollama** – download from https://ollama.com/download  
   (Available for macOS, Linux, and Windows.)

2. **Pull a model** that supports tool/function calling:
   ```bash
   ollama pull llama3.1
   ```
   Other good choices: `mistral-nemo`, `qwen2.5`, `command-r`.

3. Ollama starts automatically in the background. No extra command needed.

4. Edit `.env` so it looks like this (comment out or remove `OPENAI_API_KEY`):
   ```
   OPENAI_BASE_URL=http://localhost:11434/v1
   MODEL_NAME=llama3.1
   ```

---

#### Option C – Local AI with LM Studio (free, no API key)

[LM Studio](https://lmstudio.ai) is a desktop app that downloads and runs models locally with a graphical interface.

1. **Install LM Studio** – download from https://lmstudio.ai

2. Inside LM Studio, go to the **Discover** tab and download a model  
   (e.g. *Mistral-7B-Instruct*, *Llama-3.1-8B-Instruct*, or *Qwen2.5-7B-Instruct*).

3. Go to the **Local Server** tab and click **Start Server**.

4. Edit `.env`:
   ```
   OPENAI_BASE_URL=http://localhost:1234/v1
   MODEL_NAME=mistral-7b-instruct
   ```
   (Use the model identifier that LM Studio shows in the server tab.)

---

## 6. Running the application

```bash
python main.py
```

The startup banner shows which backend and model is active:
```
============================================================
  What Next? – Activity Recommender
  Backend : http://localhost:11434/v1
  Model   : llama3.1
  Type 'exit' or 'quit' to stop.
============================================================
```

The assistant will greet you and ask for your age and location (city + street).  
Type `exit` or `quit` to stop the session.

---

## 7. Example conversation

```
============================================================
  What Next? – Activity Recommender (powered by OpenAI)
  Type 'exit' or 'quit' to stop.
============================================================

You: Hello

[ASSISTANT RESPONSE] Generating final answer …

Assistant: Hello! I'm here to help you find something fun to do. Could you tell me your age and your location (city and street or neighbourhood)?

You: I am 28, I live in Brno, Masarykova Street

[TOOL CALL] Model is calling tool 'get_recommendations' with arguments: {"location": "Brno, Masarykova Street", "age": 28, "current_time": "14:30", "season": "winter"}
[TOOL RESULT] {"location": "Brno, Masarykova Street", "recommendations": [{"name": "Ice skating", ...}]}

[ASSISTANT RESPONSE] Generating final answer ...

Assistant: Great! Here are some activities you can enjoy right now in Brno:

  1. Ice skating - perfect for a winter afternoon!
  2. Cinema - warm up with a good film.
  3. Gym / fitness centre - great for a workout.
  4. Yoga studio - relax and stretch.
  5. Online course / self-study - invest in yourself!

Enjoy your afternoon!
```

---

## 8. Running with GitHub Copilot Agent / Remote Agent

GitHub Copilot Agent (also known as the GitHub Copilot coding agent) can run this project automatically inside a cloud sandbox. Here is what a developer or maintainer needs to know.

### What is GitHub Copilot Agent?

GitHub Copilot Agent is an AI-powered agent that can read your repository, understand the code, install dependencies, and run programs – all inside an isolated cloud environment. You do not need to install anything locally.

### Using GitHub Copilot Agent to run or modify this project

1. Open the repository on GitHub.
2. Click **Copilot** (the chat icon) and start a session.
3. Ask the agent to, for example:
   - *"Install dependencies and run main.py"*
   - *"Add a new activity to the winter catalogue"*
   - *"Explain how the tool calling loop works"*
4. The agent will set up the environment (`pip install -r requirements.txt`), read the code, and carry out your request.

### Pre-configuring the API key for GitHub Copilot Agent

GitHub Copilot Agent can read repository secrets. Add your OpenAI API key as a **Codespace / Actions secret** named `OPENAI_API_KEY` in the repository settings (`Settings > Secrets and variables > Codespaces`). The application will pick it up automatically via `python-dotenv` or environment variables.

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'openai'` | Run `pip install -r requirements.txt` |
| `EnvironmentError: OPENAI_API_KEY is not set…` | Either add your API key **or** set `OPENAI_BASE_URL` in `.env` for a local backend |
| `AuthenticationError` from OpenAI | Your API key is wrong or expired – check https://platform.openai.com/api-keys |
| `RateLimitError` | You hit the API rate limit – wait a few seconds and try again |
| `Connection refused` on `localhost:11434` | Ollama is not running – install it from https://ollama.com/download |
| `Connection refused` on `localhost:1234` | LM Studio server is not started – open LM Studio → Local Server → Start Server |
| Model does not call the tool (Ollama/LM Studio) | Use a model that supports function calling, e.g. `llama3.1`, `mistral-nemo`, `qwen2.5` |
| Python version error | Upgrade to Python 3.10 or newer |
| The assistant does not ask for location | It already has context – simply provide your city and street |

---

## 10. How to get an OpenAI API key (optional)

1. Go to https://platform.openai.com/signup and create a free account.
2. After signing in, go to https://platform.openai.com/api-keys.
3. Click **Create new secret key** and give it a name (e.g. `activity-recommender`).
4. Copy the key immediately – it will not be shown again.
5. Paste it into your `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```

> **Note:** The OpenAI API is a paid service. New accounts receive some free credits. Check https://platform.openai.com/usage for your current usage and billing.

---

## License

See [LICENSE](LICENSE).
