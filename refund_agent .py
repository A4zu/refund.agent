# ============================================================
# IMPORTS & SETUP
# ============================================================
import os
import sqlite3
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from typing import TypedDict, List, Optional, Dict, Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "GMAIL_ADDRESS",
    "GMAIL_APP_PASSWORD",
    "RECEIVER_EMAIL"
]

missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]

if missing:
    raise EnvironmentError(
        f"Aşağıdakı mühit dəyişənləri tapılmadı: {', '.join(missing)}\n"
        f"Zəhmət olmasa .env faylını düzgün konfiqurasiya edin."
    )


# ============================================================
# LLM
# ============================================================

model = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0
)


# ============================================================
# STRUCTURED OUTPUT SCHEMAS
# ============================================================

class RefundClassification(BaseModel):
    is_refund: bool = Field(
        description="Whether the customer message is related to a refund."
    )


class ReasonClassification(BaseModel):
    category: Literal["wrong_item", "defect", "other"] = Field(
        description="The category of the customer's refund reason."
    )


# Create structured-output models

refund_classifier = model.with_structured_output(
    RefundClassification
)

reason_classifier = model.with_structured_output(
    ReasonClassification
)


# ============================================================
# DATABASE
# ============================================================

def setup_database():
    """Nümunə SQLite database yaradır."""

    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_number TEXT PRIMARY KEY,
            item TEXT,
            status TEXT,
            customer_email TEXT
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO orders
        (order_number, item, status, customer_email)
        VALUES
        ('12345', 'Şlanq', 'delivered', 'musteri@example.com')
    """)

    conn.commit()
    conn.close()


setup_database()


# ============================================================
# STATE
# ============================================================

class CustomerRefundState(TypedDict):
    customer_message: str

    is_refund_related: Optional[bool]

    order_number: Optional[str]
    order_valid: Optional[bool]
    order_data: Optional[Dict[str, Any]]

    refund_reason: Optional[str]
    reason_category: Optional[str]

    photo_requested: bool
    additional_details: Optional[str]

    retry_count: int
    max_retries: int

    final_reply: Optional[str]

    messages: List[Dict[str, Any]]


# ============================================================
# NODES
# ============================================================

def read_query(state: CustomerRefundState):
    """
    Customer mesajını qəbul edir.
    """

    customer_message = state["customer_message"]

    return {}


# ------------------------------------------------------------
# CLASSIFY CUSTOMER MESSAGE
# ------------------------------------------------------------

def classify_query(state: CustomerRefundState):
    """
    LLM customer mesajının refund ilə əlaqəli olub-olmadığını
    structured output vasitəsilə müəyyən edir.
    """

    customer_message = state["customer_message"]

    prompt = f"""
You are a customer service classification agent.

Determine whether the customer's message is related to a refund.

Return:
- is_refund = true if the customer is asking for a refund,
  returning an item, getting money back, or reporting an issue
  that is clearly related to a refund.
- is_refund = false if the message is unrelated to refunds.

Customer message:
{customer_message}
"""

    result = refund_classifier.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "is_refund_related": result.is_refund,

        "messages": state.get("messages", []) + [
            {
                "role": "user",
                "content": customer_message
            },
            {
                "role": "assistant",
                "content": f"is_refund={result.is_refund}"
            }
        ]
    }


# ------------------------------------------------------------
# NORMAL REPLY
# ------------------------------------------------------------

def normal_reply(state: CustomerRefundState):
    """
    Refund ilə əlaqəli olmayan mesajlara cavab verir.
    """

    reply = (
        "This issue is not related to refunds. "
        "Sorry, we cannot help you with this request."
    )

    return {
        "final_reply": reply
    }


# ------------------------------------------------------------
# ASK ORDER NUMBER
# ------------------------------------------------------------

def ask_order_number(state: CustomerRefundState):

    order_number = input(
        "Please tell us your order number: "
    )

    return {
        "order_number": order_number
    }


# ------------------------------------------------------------
# CHECK ORDER
# ------------------------------------------------------------

def check_order(state: CustomerRefundState):
    """
    Order number-ı SQLite database-də yoxlayır.
    """

    order_number = state.get("order_number")

    order_data = None

    try:

        conn = sqlite3.connect("orders.db")
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM orders
            WHERE order_number = ?
            """,
            (order_number,)
        )

        row = cursor.fetchone()

        if row:
            order_data = dict(row)

        conn.close()

    except sqlite3.Error as e:

        print("Database Error:", e)

        order_data = None

    return {
        "order_valid": order_data is not None,
        "order_data": order_data
    }


# ------------------------------------------------------------
# ASK AGAIN
# ------------------------------------------------------------

def ask_again(state: CustomerRefundState):
    """
    Order tapılmadıqda yenidən order number istəyir.
    """

    retry_count = state.get("retry_count", 0) + 1

    if retry_count >= state.get("max_retries", 5):

        return {
            "retry_count": retry_count,
            "final_reply": (
                "We were unable to verify your order number. "
                "Unfortunately, we cannot proceed with your refund request."
            )
        }

    order_number = input(
        "Order not found. "
        "Please enter a valid order number: "
    )

    return {
        "order_number": order_number,
        "retry_count": retry_count
    }


# ------------------------------------------------------------
# ASK REASON
# ------------------------------------------------------------

def ask_reason(state: CustomerRefundState):
    """
    Refund səbəbini customer-dan alır.
    """

    reason = input(
        "Please tell us the reason for the refund: "
    )

    return {
        "refund_reason": reason
    }


# ------------------------------------------------------------
# CLASSIFY REASON
# ------------------------------------------------------------

def classify_reason(state: CustomerRefundState):
    """
    Refund səbəbini structured output ilə
    üç kateqoriyadan birinə ayırır.
    """

    reason = state["refund_reason"]

    prompt = f"""
You are a customer service refund classification agent.

Classify the customer's refund reason into exactly one
of these categories:

1. wrong_item
   The customer received the wrong product.

2. defect
   The product is damaged, defective, broken,
   or does not work properly.

3. other
   Any refund reason that does not belong to the
   previous two categories.

Customer's reason:
{reason}
"""

    result = reason_classifier.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "reason_category": result.category,

        "messages": state.get("messages", []) + [
            {
                "role": "user",
                "content": reason
            },
            {
                "role": "assistant",
                "content": f"category={result.category}"
            }
        ]
    }


# ------------------------------------------------------------
# REQUEST PHOTO
# ------------------------------------------------------------

def request_photo(state: CustomerRefundState):
    """
    Prototype version:
    Real image upload əvəzinə customer-dan
    əlavə məlumat alır.
    """

    photo_info = input(
        "Please describe the photo of the item "
        "or provide additional information: "
    )

    return {
        "photo_requested": True,
        "additional_details": photo_info
    }


# ------------------------------------------------------------
# ASK MORE DETAILS
# ------------------------------------------------------------

def ask_more_details(state: CustomerRefundState):

    details = input(
        "Please tell us more detailed information: "
    )

    return {
        "additional_details": details
    }


# ------------------------------------------------------------
# HUMAN REVIEW
# ------------------------------------------------------------

def human_review(state: CustomerRefundState):
    """
    Refund request-i human reviewer-ə
    SMTP vasitəsilə email olaraq göndərir.
    """

    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    subject = (
        f"New Refund Request - "
        f"Order #{state.get('order_number', 'N/A')}"
    )

    body = f"""
A new refund request needs human review.

Customer message:
{state.get('customer_message')}

Customer email:
{state.get('order_data', {}).get('customer_email')}

Order number:
{state.get('order_number')}

Order data:
{state.get('order_data')}

Refund reason:
{state.get('refund_reason')}

Reason category:
{state.get('reason_category')}

Additional details:
{state.get('additional_details')}

Photo requested:
{state.get('photo_requested')}
"""

    msg = MIMEMultipart()

    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(
        MIMEText(body, "plain")
    )

    try:

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            sender_email,
            sender_password
        )

        server.sendmail(
            sender_email,
            receiver_email,
            msg.as_string()
        )

        server.quit()

        print(
            f"Refund request for order "
            f"{state.get('order_number')} "
            f"sent to {receiver_email}"
        )

        return {
            "email_sent": True
        }

    except Exception as e:

        print(
            "Email sending failed:",
            e
        )

        return {
            "email_sent": False
        }


# ------------------------------------------------------------
# NOTIFY CUSTOMER
# ------------------------------------------------------------

def notify_customer(state: CustomerRefundState):

    reply = (
        "Thank you! Your refund request has been "
        "sent for human review. "
        "We will inform you within the next 7 days."
    )

    return {
        "final_reply": reply
    }


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_by_intent(state: CustomerRefundState) -> str:
    """
    Refund olub-olmamasına əsasən
    növbəti node-u müəyyən edir.
    """

    if state["is_refund_related"]:
        return "refund_related"

    return "not_refund_related"


def route_by_order_validity(state: CustomerRefundState) -> str:
    """
    Order-in database-də mövcud olub-olmamasına
    əsasən növbəti node-u müəyyən edir.
    """

    # Retry limitinə çatılıbsa workflow-u bitiririk.
    if state.get("final_reply"):
        return "retry_limit_reached"

    if state["order_valid"]:
        return "valid"

    return "invalid"


def route_by_reason(state: CustomerRefundState) -> str:
    """
    Refund reason category-yə əsasən
    növbəti node-u müəyyən edir.
    """

    category = state.get("reason_category")

    if category in ["wrong_item", "defect"]:
        return "request_photo"

    return "ask_more_details"


def route_after_human_review(state: CustomerRefundState) -> str:
    """
    Email göndərilibsə customer notification-a,
    göndərilməyibsə error reply-a yönləndirir.
    """

    if state.get("email_sent"):
        return "success"

    return "email_failed"


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(CustomerRefundState)


# ------------------------------------------------------------
# ADD NODES
# ------------------------------------------------------------

builder.add_node(
    "read_query",
    read_query
)

builder.add_node(
    "classify_query",
    classify_query
)

builder.add_node(
    "normal_reply",
    normal_reply
)

builder.add_node(
    "ask_order_number",
    ask_order_number
)

builder.add_node(
    "check_order",
    check_order
)

builder.add_node(
    "ask_again",
    ask_again
)

builder.add_node(
    "ask_reason",
    ask_reason
)

builder.add_node(
    "classify_reason",
    classify_reason
)

builder.add_node(
    "request_photo",
    request_photo
)

builder.add_node(
    "ask_more_details",
    ask_more_details
)

builder.add_node(
    "human_review",
    human_review
)

builder.add_node(
    "notify_customer",
    notify_customer
)


# ============================================================
# FIXED EDGES
# ============================================================

builder.add_edge(
    START,
    "read_query"
)

builder.add_edge(
    "read_query",
    "classify_query"
)

builder.add_edge(
    "ask_order_number",
    "check_order"
)

builder.add_edge(
    "ask_again",
    "check_order"
)

builder.add_edge(
    "ask_reason",
    "classify_reason"
)

builder.add_edge(
    "request_photo",
    "human_review"
)

builder.add_edge(
    "ask_more_details",
    "human_review"
)

builder.add_edge(
    "normal_reply",
    END
)

builder.add_edge(
    "notify_customer",
    END
)


# ============================================================
# CONDITIONAL EDGES
# ============================================================

# ------------------------------------------------------------
# 1. REFUND OR NOT?
# ------------------------------------------------------------

builder.add_conditional_edges(
    "classify_query",
    route_by_intent,
    {
        "refund_related": "ask_order_number",
        "not_refund_related": "normal_reply"
    }
)


# ------------------------------------------------------------
# 2. ORDER VALID OR INVALID?
# ------------------------------------------------------------

builder.add_conditional_edges(
    "check_order",
    route_by_order_validity,
    {
        "valid": "ask_reason",
        "invalid": "ask_again",
        "retry_limit_reached": END
    }
)


# ------------------------------------------------------------
# 3. REFUND REASON
# ------------------------------------------------------------

builder.add_conditional_edges(
    "classify_reason",
    route_by_reason,
    {
        "request_photo": "request_photo",
        "ask_more_details": "ask_more_details"
    }
)


# ============================================================
# COMPILE
# ============================================================

graph = builder.compile()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    refund_message = {
        "customer_message":
            "I received the wrong item and want a refund.",

        "is_refund_related": None,

        "order_number": None,

        "order_valid": None,

        "order_data": None,

        "refund_reason": None,

        "reason_category": None,

        "photo_requested": False,

        "additional_details": None,

        "retry_count": 0,

        "max_retries": 5,

        "final_reply": None,

        "messages": []
    }

    result = graph.invoke(
        refund_message
    )

    print("\nFINAL RESPONSE:")
    print(result.get("final_reply"))
