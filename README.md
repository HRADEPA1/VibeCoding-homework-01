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
6. [Running the application](#6-running-the-application)
7. [Example conversation](#7-example-conversation)
8. [Running with GitHub Codex / Remote Agent](#8-running-with-github-codex--remote-agent)
9. [Troubleshooting](#9-troubleshooting)
10. [How to get an OpenAI API key](#10-how-to-get-an-openai-api-key)

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

| What you need | Minimum version | Where to get it |
|---|---|---|
| **Python** | 3.10+ | https://www.python.org/downloads/ |
| **pip** | bundled with Python | — |
| **OpenAI API key** | — | https://platform.openai.com/api-keys |
| **Internet connection** | — | required for the OpenAI API |

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
- **openai** – the official OpenAI Python SDK.
- **python-dotenv** – reads the `.env` file so your API key stays out of the code.

### Step 4 – Add your OpenAI API key

1. Copy the example file:
   ```bash
   # macOS / Linux
   cp .env.example .env

   # Windows (Command Prompt)
   copy .env.example .env
   ```
2. Open `.env` with any text editor (Notepad, VS Code, etc.).
3. Replace `sk-...your-key-here...` with your real API key (see [section 10](#10-how-to-get-an-openai-api-key)).
4. Save the file.

> ⚠️ **Never share your `.env` file or commit it to GitHub.** It is already listed in `.gitignore`.

---

## 6. Running the application

```bash
python main.py
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

## 8. Running with GitHub Codex / Remote Agent

GitHub Codex (also known as the GitHub Copilot coding agent) can run this project automatically inside a cloud sandbox. Here is what a developer or maintainer needs to know.

### What is Codex?

Codex is an AI-powered agent that can read your repository, understand the code, install dependencies, and run programs – all inside an isolated cloud environment. You do not need to install anything locally.

### Using Codex to run or modify this project

1. Open the repository on GitHub.
2. Click **Copilot** (the chat icon) and start a session.
3. Ask Codex to, for example:
   - *"Install dependencies and run main.py"*
   - *"Add a new activity to the winter catalogue"*
   - *"Explain how the tool calling loop works"*
4. Codex will set up the environment (`pip install -r requirements.txt`), read the code, and carry out your request.

### Pre-configuring the API key for Codex

Codex can read repository secrets. Add your OpenAI API key as a **Codespace / Actions secret** named `OPENAI_API_KEY` in the repository settings (`Settings > Secrets and variables > Codespaces`). The application will pick it up automatically via `python-dotenv` or environment variables.

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'openai'` | Run `pip install -r requirements.txt` |
| `EnvironmentError: OPENAI_API_KEY is not set` | Make sure you created `.env` from `.env.example` and added your key |
| `AuthenticationError` from OpenAI | Your API key is wrong or has expired – check https://platform.openai.com/api-keys |
| `RateLimitError` | You have hit the API rate limit – wait a few seconds and try again |
| Python version error (`match` or walrus syntax) | Upgrade to Python 3.10 or newer |
| The assistant does not ask for location | It already has context – simply provide your city and street |

---

## 10. How to get an OpenAI API key

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
