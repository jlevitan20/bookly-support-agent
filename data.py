"""
data.py — Mock data and tool functions for the Bookly support agent.

This file contains:
1. Mock customers, orders, and policy articles (the "database")
2. Tool functions the agent calls to look up data and take actions

In production, these functions would hit a real database or API.
The agent doesn't know the difference — it just calls the function and gets a result.
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# MOCK DATA
# ---------------------------------------------------------------------------

# Current date for calculating return eligibility, etc.
TODAY = datetime.now()

# Customers — keyed by email (the unique identifier)
CUSTOMERS = {
    "jane.smith@email.com": {
        "customer_id": "CUST-001",
        "name": "Jane Smith",
        "email": "jane.smith@email.com",
    },
    "marcus.jones@email.com": {
        "customer_id": "CUST-002",
        "name": "Marcus Jones",
        "email": "marcus.jones@email.com",
    },
    "sarah.chen@email.com": {
        "customer_id": "CUST-003",
        "name": "Sarah Chen",
        "email": "sarah.chen@email.com",
    },
    "tom.baker@email.com": {
        "customer_id": "CUST-004",
        "name": "Tom Baker",
        "email": "tom.baker@email.com",
    },
}

# Orders — keyed by order ID
# Each order is designed to showcase a specific agent behavior in the demo
ORDERS = {
    # Jane's orders — she has two, which forces the "which order?" flow
    "ORD-7201": {
        "order_id": "ORD-7201",
        "customer_email": "jane.smith@email.com",
        "customer_name": "Jane Smith",
        "status": "in_transit",
        "items": [
            {"title": "Dune by Frank Herbert", "format": "Hardcover", "price": 18.99},
            {"title": "Project Hail Mary by Andy Weir", "format": "Paperback", "price": 14.99},
        ],
        "total": 33.98,
        "order_date": (TODAY - timedelta(days=3)).strftime("%B %d, %Y"),
        "estimated_delivery": (TODAY + timedelta(days=4)).strftime("%B %d, %Y"),
        "tracking_number": "1Z999AA10123456784",
        "shipping_method": "Standard (5-7 business days)",
        "returned": False,
    },
    "ORD-7145": {
        "order_id": "ORD-7145",
        "customer_email": "jane.smith@email.com",
        "customer_name": "Jane Smith",
        "status": "delivered",
        "items": [
            {"title": "The Great Gatsby by F. Scott Fitzgerald", "format": "Paperback", "price": 12.99},
        ],
        "total": 12.99,
        "order_date": (TODAY - timedelta(days=18)).strftime("%B %d, %Y"),
        "delivered_date": (TODAY - timedelta(days=12)).strftime("%B %d, %Y"),
        "shipping_method": "Standard (5-7 business days)",
        "returned": False,
    },
    # Marcus — delivered 45 days ago, PAST the 30-day return window
    "ORD-6890": {
        "order_id": "ORD-6890",
        "customer_email": "marcus.jones@email.com",
        "customer_name": "Marcus Jones",
        "status": "delivered",
        "items": [
            {"title": "Atomic Habits by James Clear", "format": "Hardcover", "price": 22.99},
            {"title": "Deep Work by Cal Newport", "format": "Paperback", "price": 15.99},
        ],
        "total": 38.98,
        "order_date": (TODAY - timedelta(days=52)).strftime("%B %d, %Y"),
        "delivered_date": (TODAY - timedelta(days=45)).strftime("%B %d, %Y"),
        "shipping_method": "Expedited (2-3 business days)",
        "returned": False,
    },
    # Sarah — digital purchase (ebook), non-refundable per policy
    "ORD-7300": {
        "order_id": "ORD-7300",
        "customer_email": "sarah.chen@email.com",
        "customer_name": "Sarah Chen",
        "status": "delivered",
        "items": [
            {"title": "Educated by Tara Westover", "format": "Ebook", "price": 11.99},
        ],
        "total": 11.99,
        "order_date": (TODAY - timedelta(days=5)).strftime("%B %d, %Y"),
        "delivered_date": (TODAY - timedelta(days=5)).strftime("%B %d, %Y"),
        "shipping_method": "Digital delivery (instant)",
        "returned": False,
    },
    # Tom — order still processing, hasn't shipped yet (cancellation eligible)
    "ORD-7350": {
        "order_id": "ORD-7350",
        "customer_email": "tom.baker@email.com",
        "customer_name": "Tom Baker",
        "status": "processing",
        "items": [
            {"title": "The Midnight Library by Matt Haig", "format": "Hardcover", "price": 17.99},
        ],
        "total": 17.99,
        "order_date": TODAY.strftime("%B %d, %Y"),
        "estimated_ship_date": (TODAY + timedelta(days=1)).strftime("%B %d, %Y"),
        "shipping_method": "Standard (5-7 business days)",
        "returned": False,
    },
    # Tom — already returned order
    "ORD-6500": {
        "order_id": "ORD-6500",
        "customer_email": "tom.baker@email.com",
        "customer_name": "Tom Baker",
        "status": "returned",
        "items": [
            {"title": "Sapiens by Yuval Noah Harari", "format": "Paperback", "price": 16.99},
        ],
        "total": 16.99,
        "order_date": (TODAY - timedelta(days=40)).strftime("%B %d, %Y"),
        "delivered_date": (TODAY - timedelta(days=33)).strftime("%B %d, %Y"),
        "returned": True,
        "return_date": (TODAY - timedelta(days=28)).strftime("%B %d, %Y"),
        "refund_amount": 16.99,
        "shipping_method": "Standard (5-7 business days)",
    },
}

# Policy articles — the agent's knowledge base
# Structured so the agent retrieves and CITES specific articles
# instead of making up answers from general knowledge
POLICIES = {
    "returns": {
        "id": "POL-001",
        "title": "Return Policy",
        "content": (
            "Customers may return physical items within 30 days of delivery. "
            "Items must be in original condition with packaging intact. "
            "Digital purchases (ebooks, audiobooks) are non-refundable. "
            "To initiate a return, customers need their order ID and the reason for return. "
            "Eligible return reasons: changed my mind, wrong item received, "
            "item damaged on arrival, item not as described. "
            "Refunds are issued to the original payment method within 5-10 business days. "
            "Alternatively, customers may choose instant store credit."
        ),
    },
    "shipping": {
        "id": "POL-002",
        "title": "Shipping Policy",
        "content": (
            "Standard shipping: 5-7 business days, free on orders over $35. "
            "Expedited shipping: 2-3 business days, $5.99. "
            "Orders are processed and shipped within 1-2 business days of placement. "
            "Digital items are delivered instantly to the customer's Bookly account. "
            "Tracking information is emailed once the order ships. "
            "Bookly ships to all 50 US states. International shipping is not available at this time."
        ),
    },
    "refunds": {
        "id": "POL-003",
        "title": "Refund Policy",
        "content": (
            "Refunds for returned items are processed within 5-10 business days "
            "after we receive the returned item. Refunds are issued to the original "
            "payment method. Alternatively, customers can choose store credit, "
            "which is applied instantly. Shipping costs are non-refundable. "
            "For damaged items, we offer a full refund including original shipping costs."
        ),
    },
    "cancellation": {
        "id": "POL-004",
        "title": "Order Cancellation Policy",
        "content": (
            "Orders can be cancelled if they have not yet shipped. "
            "Once an order has shipped, it cannot be cancelled — the customer "
            "should wait for delivery and then initiate a return. "
            "Cancelled orders are refunded in full within 3-5 business days."
        ),
    },
    "account": {
        "id": "POL-005",
        "title": "Account & Password Policy",
        "content": (
            "Customers can reset their password via the 'Forgot Password' link on the login page. "
            "A password reset link is sent to the email address on file. "
            "For security, support agents cannot change passwords or email addresses directly. "
            "If a customer cannot access their email, they should contact support for "
            "identity verification and manual account recovery."
        ),
    },
    "damaged_items": {
        "id": "POL-006",
        "title": "Damaged Item Policy",
        "content": (
            "If an item arrives damaged, the customer is eligible for a full refund "
            "or a free replacement regardless of the standard return window. "
            "Customers may be asked to provide a photo of the damage, but this is "
            "not required to process the claim. Damaged item claims are prioritized "
            "and typically resolved within 1-2 business days."
        ),
    },
}


# ---------------------------------------------------------------------------
# TOOL FUNCTIONS — called by the agent via OpenAI function calling
# ---------------------------------------------------------------------------

def verify_customer_email(email: str) -> dict:
    """
    Verify that a customer exists by their email address.
    This is the first step before the agent can access any order data.
    Returns customer info if found, or an error if not.
    """
    email = email.lower().strip()
    customer = CUSTOMERS.get(email)
    if customer:
        # Find how many orders this customer has
        order_count = sum(
            1 for o in ORDERS.values() if o["customer_email"] == email
        )
        return {
            "verified": True,
            "customer_name": customer["name"],
            "customer_id": customer["customer_id"],
            "email": customer["email"],
            "order_count": order_count,
        }
    return {
        "verified": False,
        "message": f"No account found for {email}. Please double-check the email address.",
    }


def lookup_order(order_id: str) -> dict:
    """
    Look up a specific order by its order ID.
    Returns full order details or an error if not found.
    """
    order_id = order_id.upper().strip()
    order = ORDERS.get(order_id)
    if order:
        return order
    return {
        "error": True,
        "message": f"No order found with ID {order_id}. Order IDs look like ORD-XXXX.",
    }


def search_orders_by_email(email: str) -> dict:
    """
    Find all orders for a customer by email.
    Useful when the customer doesn't have their order ID handy.
    """
    email = email.lower().strip()
    matching_orders = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "order_date": o["order_date"],
            "total": o["total"],
            "items": [
                f"{item['title']} (Order {o['order_id']}, Status: {o['status']})"
                for item in o["items"]
            ],
        }
        for o in ORDERS.values()
        if o["customer_email"] == email
    ]
    if matching_orders:
        return {"orders": matching_orders, "count": len(matching_orders)}
    return {"orders": [], "count": 0, "message": "No orders found for this email."}


def initiate_return(order_id: str, reason: str, item_title: str = "") -> dict:
    """
    Attempt to initiate a return for an order.
    Checks eligibility: 30-day window, physical item, not already returned.
    Returns approval or denial with explanation.
    """
    order_id = order_id.upper().strip()
    order = ORDERS.get(order_id)

    if not order:
        return {"approved": False, "reason": f"Order {order_id} not found."}

    # Auto-correct: if item_title provided but isn't in this order, find the right one
    if item_title:
        item_in_order = any(item_title.lower() in item["title"].lower() for item in order["items"])
        if not item_in_order:
            customer_email = order.get("customer_email", "")
            for oid, o in ORDERS.items():
                if o.get("customer_email") == customer_email and oid != order_id:
                    if any(item_title.lower() in item["title"].lower() for item in o["items"]):
                        order_id = oid
                        order = o
                        break

    # Already returned?
    if order.get("returned"):
        return {
            "approved": False,
            "reason": f"Order {order_id} was already returned on {order.get('return_date', 'a previous date')}.",
        }

    # Not delivered yet?
    if order["status"] in ("processing", "in_transit"):
        return {
            "approved": False,
            "reason": f"Order {order_id} hasn't been delivered yet (status: {order['status']}). "
                      "If you'd like to cancel instead, I can help with that.",
        }

    # Digital item?
    has_digital = any(item["format"] == "Ebook" for item in order["items"])
    if has_digital:
        return {
            "approved": False,
            "reason": "This order contains a digital item (ebook). Per our policy, "
                      "digital purchases are non-refundable.",
            "policy_reference": "POL-001 — Return Policy",
        }

    # Past 30-day return window?
    if order.get("delivered_date"):
        delivered = datetime.strptime(order["delivered_date"], "%B %d, %Y")
        days_since = (TODAY - delivered).days
        if days_since > 30:
            return {
                "approved": False,
                "reason": f"This order was delivered {days_since} days ago, which is past our "
                          f"30-day return window. The return window closed on "
                          f"{(delivered + timedelta(days=30)).strftime('%B %d, %Y')}.",
                "policy_reference": "POL-001 — Return Policy",
            }

    # All checks passed — approve the return
    return {
        "approved": True,
        "return_id": f"RET-{order_id.split('-')[1]}",
        "order_id": order_id,
        "reason": reason,
        "refund_amount": order["total"],
        "message": (
            f"Return approved for order {order_id}. A prepaid return label will be emailed "
            f"to the customer. Refund of ${order['total']:.2f} will be processed within "
            f"5-10 business days after we receive the item, or instant store credit is available."
        ),
    }


def search_knowledge_base(query: str) -> dict:
    """
    Search the policy knowledge base by keyword.
    Returns matching policy articles for the agent to cite.
    """
    query_lower = query.lower()

    # Simple keyword matching — in production this would be vector search / RAG
    keyword_map = {
        "returns": ["return", "send back", "refund", "exchange", "return policy"],
        "shipping": ["shipping", "delivery", "ship", "tracking", "how long", "arrive"],
        "refunds": ["refund", "money back", "reimburse", "credit"],
        "cancellation": ["cancel", "cancellation", "stop order"],
        "account": ["password", "account", "login", "sign in", "reset", "email"],
        "damaged_items": ["damaged", "broken", "defective", "ripped", "torn"],
    }

    matches = []
    for policy_key, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            policy = POLICIES[policy_key]
            matches.append({
                "policy_id": policy["id"],
                "title": policy["title"],
                "content": policy["content"],
            })

    if matches:
        return {"results": matches, "count": len(matches)}
    return {
        "results": [],
        "count": 0,
        "message": "No matching policies found. The agent should let the customer know "
                   "and offer to connect them with a human agent.",
    }


def escalate_to_human(summary: str, reason: str, customer_sentiment: str) -> dict:
    """
    Escalate the conversation to a human agent.
    Generates a structured handoff packet with full context.
    """
    return {
        "escalated": True,
        "ticket_id": "ESC-" + datetime.now().strftime("%Y%m%d%H%M"),
        "handoff_summary": summary,
        "escalation_reason": reason,
        "customer_sentiment": customer_sentiment,
        "message": "This conversation has been escalated to a human support agent. "
                   "The customer should expect a response within 2-4 hours during business hours.",
    }


# ---------------------------------------------------------------------------
# TOOL REGISTRY — maps function names to their implementations
# Used by the agent to dispatch tool calls from OpenAI
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "verify_customer_email": verify_customer_email,
    "lookup_order": lookup_order,
    "search_orders_by_email": search_orders_by_email,
    "initiate_return": initiate_return,
    "search_knowledge_base": search_knowledge_base,
    "escalate_to_human": escalate_to_human,
}
