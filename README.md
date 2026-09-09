# Bookly Support Agent

An AI-powered customer support agent for Bookly, a fictional online bookstore. Built with Python, FastAPI, and OpenAI's function-calling API.

## Architecture

```
Customer → Chat UI → FastAPI Server → OpenAI (gpt-4o-mini)
                                           ↕
                                      Tool Functions
                                    (verify, lookup, return,
                                     knowledge base, escalate)
                                           ↕
                                      Mock Data Layer
                                   (customers, orders, policies)
```

**Key design decisions:**
- **Structured tool use** — the LLM decides *what* to do, but tools enforce *how*. The agent can't invent order data because it must call `lookup_order()` to get it.
- **Verification-first flow** — customer email is verified before any order data is shared, preventing unauthorized access.
- **Knowledge base with citation** — policy questions are answered by retrieving specific policy articles, not from the LLM's general knowledge. This prevents hallucinated policies.
- **Conversation state tracking** — a structured state object tracks verification status, current order, and actions taken, injected into each prompt so the agent knows where it is.
- **Graceful escalation** — when the agent can't resolve an issue, it generates a structured handoff packet for a human agent with full context and customer sentiment.

## Quick Start

### Prerequisites
- Python 3.11+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd bookly-support-agent

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Run the server
uvicorn server:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Demo Scenarios

Try these conversations to see the agent's key behaviors:

**1. Order status with verification (multi-turn + tool use)**
> "Where's my order?"
> → Agent asks for email → provide `jane.smith@email.com`
> → Agent verifies, finds 2 orders, asks which one
> → Provide `ORD-7201` → Agent shows tracking info

**2. Return flow with denial (clarifying question + policy citation)**
> "I want to return a book"
> → Agent asks for email → provide `marcus.jones@email.com`
> → Agent asks for order ID → provide `ORD-6890`
> → Agent checks eligibility → denies (past 30-day window) with policy citation
> → Push back → Agent offers escalation to human

**3. Policy question (knowledge base citation)**
> "What's your return policy?"
> → Agent searches knowledge base and cites the specific policy

**4. Digital item return (edge case)**
> Use email `sarah.chen@email.com`, order `ORD-7300`
> → Agent denies return for ebook with policy citation

## File Structure

```
bookly-support-agent/
├── server.py       # FastAPI web server — routes and API endpoint
├── agent.py        # Agent logic — system prompt, tool definitions, conversation loop
├── data.py         # Mock data (customers, orders, policies) and tool functions
├── static/
│   └── index.html  # Chat UI — single-page web interface
├── requirements.txt
└── README.md
```

## Mock Data

The agent works with a small set of realistic mock data designed to showcase specific behaviors:

| Order ID | Customer | Status | Demo Purpose |
|----------|----------|--------|-------------|
| ORD-7201 | Jane Smith | In Transit | Happy path — order tracking |
| ORD-7145 | Jane Smith | Delivered (12 days ago) | Eligible for return |
| ORD-6890 | Marcus Jones | Delivered (45 days ago) | Past return window — denial |
| ORD-7300 | Sarah Chen | Delivered (Ebook) | Digital item — non-refundable |
| ORD-7350 | Tom Baker | Processing | Cancellation eligible |
| ORD-6500 | Tom Baker | Returned | Already returned — edge case |

## Built With

- **Python** + **FastAPI** — lightweight, fast web server
- **OpenAI gpt-4o-mini** — LLM with function-calling for tool use
- **Vanilla HTML/CSS/JS** — no frontend framework, just a clean chat interface
