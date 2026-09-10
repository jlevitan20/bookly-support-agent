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

**Design decisions**

- The model decides what to do. The tools decide how it happens. The agent can't invent order data, because the only way to get it is to call `lookup_order()`.
- Nothing about an order is shared until the customer's email is verified.
- Policy questions are answered by retrieving the policy article and citing it. The model never answers from general knowledge, so it can't make up a refund window.
- A state object tracks verification, the current order, and actions taken, and goes into the prompt on every turn. It also keeps an issue ledger: each thing the customer asked for, with a status (open, resolved, escalated) set by tool results, never by the model saying it helped. A message with two requests can't quietly lose the second one.
- When the agent can't resolve something, it hands off to a person with a summary, the reason, and the customer's mood.

**The guardrails live in code.** Before any tool runs, `_execute_tool()` in `agent.py` checks that the
customer is verified, that the order ID is real, that the order belongs to this customer, that a return
reason is one the customer actually gave, and that a destructive action was confirmed on a later turn. The
order search always uses the verified customer's email, whatever the model passes, so it can't be pointed at
another account. A jailbroken prompt can change what the model says. It can't change what the server allows.

### Known tradeoffs

- **Conversation state round-trips through the client.** The server is stateless. `conversation_state` goes
  to the browser and comes back with the next message. That keeps the demo easy to run, but a crafted
  request could claim `verified: true`. In production the state moves to a server-side session store keyed
  by conversation ID. The guardrail logic doesn't change, only where the state lives.
- **Keyword search stands in for retrieval.** `search_knowledge_base` matches keywords against six policy
  articles. A real policy corpus needs vector search. The rule that matters, retrieve and cite instead of
  recalling, is already enforced.
- **Mock data is in-process.** Returns change nothing durable. Restarting the server resets everything.
- **Issues are keyed by type.** The ledger opens one issue per intent (return, order status, policy,
  account), so two different order questions in one conversation share a single `order_inquiry` issue.
  Keying by `(intent, order_id)` is the production follow-up.

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

Try these conversations to see the main behaviors:

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
├── server.py       # FastAPI server: routes and the streaming endpoint
├── agent.py        # Agent logic: system prompt, tool definitions, conversation loop
├── data.py         # Mock data (customers, orders, policies) and tool functions
├── static/
│   └── index.html  # Chat UI, a single page
├── requirements.txt
├── .env.example    # Copy to .env and add your OpenAI key
└── README.md
```

## Mock Data

The mock data is small, and each order exists to trigger one specific behavior:

| Order ID | Customer | Status | Demo Purpose |
|----------|----------|--------|-------------|
| ORD-7201 | Jane Smith | In Transit | Happy path, order tracking |
| ORD-7145 | Jane Smith | Delivered (12 days ago) | Eligible for return |
| ORD-6890 | Marcus Jones | Delivered (45 days ago) | Past return window, denied |
| ORD-7300 | Sarah Chen | Delivered (Ebook) | Digital item, not refundable |
| ORD-7350 | Tom Baker | Processing | Cancellation eligible |
| ORD-6500 | Tom Baker | Returned | Already returned |

## Built With

- Python and FastAPI for the server
- OpenAI gpt-4o-mini with function calling for tool use
- Plain HTML, CSS, and JavaScript for the chat UI. No frontend framework.
