![Image](assets/image.png)
# Refund Agent (LangGraph)

A simple AI agent that analyzes customer messages to determine whether they're
a refund request, verifies the order number, gathers the reason and details,
and emails a human reviewer for approval. Built with
[LangGraph](https://langchain-ai.github.io/langgraph/) and an OpenAI model.

## How it works

```
START → read_query → classify_query ──┬─ not refund → normal_reply → END
                                       │
                                       └─ refund → ask_order_number → check_order
                                                                          │
                                                        ┌─ invalid ───────┘
                                                        │        (ask_again → recheck)
                                                        └─ valid → ask_reason → classify_reason
                                                                                      │
                                                              ┌── wrong_item/defect ──┤
                                                              │                       └── other
                                                       request_photo            ask_more_details
                                                              │                       │
                                                              └────→ human_review ←───┘
                                                                          │
                                                                   notify_customer → END
```

At the `human_review` step, all details of the request are emailed to a
human reviewer (`RECEIVER_EMAIL`).

## Requirements

- Python 3.10+
- An OpenAI API key
- A Gmail account + an **App Password** (not your regular password)

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-link>
cd refund-agent

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env
```

Then open `.env` and fill in your own values:

```
OPENAI_API_KEY=sk-...
GMAIL_ADDRESS=your-sending-account@gmail.com
GMAIL_APP_PASSWORD=your-16-char-app-password
RECEIVER_EMAIL=reviewer@example.com
```

### How to get a Gmail App Password

1. Enable 2-Step Verification on your Gmail account.
2. Go to https://myaccount.google.com/apppasswords
3. Create a new App Password (e.g. name it "Refund Agent").
4. Copy the 16-character code with **no spaces** into `GMAIL_APP_PASSWORD`
   in your `.env` file.

> ⚠️ Do not use your regular Gmail password here — it won't work and isn't secure.

## Running

```bash
python refund_agent.py
```

The script will prompt you in the console (via `input()`) for the order
number, reason, etc. A sample order is preloaded in the database for testing:

- Order number: `12345`

## File structure

```
refund-agent/
├── refund_agent.py     # Main agent code (LangGraph graph)
├── requirements.txt    # Python dependencies
├── .env.example         # Example environment variables (copy and fill in)
├── .gitignore           # Keeps .env, etc. out of Git
└── README.md
```

## Security notes

- **Never** commit your `.env` file or any keys/passwords to Git — `.gitignore`
  already excludes it, but double-check before pushing.
- If you're sharing this project as a **public** GitHub repo, make sure only
  `.env.example` gets uploaded, not `.env`.
- `orders.db` is created automatically on first run, so it's excluded from Git.

## Pushing to GitHub (first time)

```bash
cd refund-agent
git init
git add .
git status   # make sure .env is NOT listed!
git commit -m "Initial commit: refund agent"
git branch -M main
git remote add origin https://github.com/<your-username>/refund-agent.git
git push -u origin main
```

## License

Free to use — feel free to modify this project as you like.
