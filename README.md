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

**Guardrails are enforced server-side, not by the prompt.** Before any tool runs, `_execute_tool()` in
`agent.py` independently checks that the customer is verified, that the order ID is well-formed and real,
that the order belongs to *this* customer, and that a destructive action has been explicitly confirmed. The
order search is scoped to the verified customer's email regardless of what the model passes, so it cannot be
aimed at another account. A jailbroken prompt still can't get past these — the model proposes, the server
disposes.

### Known tradeoffs

- **Conversation state round-trips through the client.** The server is stateless; `conversation_state` is
  returned to the browser and sent back each turn. That keeps the demo trivially runnable, but it means a
  crafted request could assert `verified: true`. In production this moves to a server-side session store
  keyed by conversation ID — the guardrail logic is unchanged, only where state lives.
- **Keyword search stands in for retrieval.** `search_knowledge_base` matches keywords against six policy
  articles. With a real policy corpus this becomes vector search, but the important property — the agent
  must retrieve and cite rather than recall — is already enforced.
- **Mock data is in-process.** Returns mutate nothing durable; restarting the server resets everything.

## Quick Start

### Prerequisites
- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

```bash
# Clone the repo
git clone https://github.com/jlevitan20/bookly-support-agent.git
cd bookly-support-agent

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

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
├── .env.example    # Copy to .env and add your OpenAI key
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
