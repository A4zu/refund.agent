# ============================================================
# IMPORTS & SETUP
# ============================================================
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# .env faylından mühit dəyişənlərini yüklə
load_dotenv()

# Tələb olunan mühit dəyişənlərini yoxla
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "RECEIVER_EMAIL"]
missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if missing:
    raise EnvironmentError(
        f"Aşağıdakı mühit dəyişənləri tapılmadı: {', '.join(missing)}\n"
        f"Zəhmət olmasa .env.example faylını .env adı ilə kopyalayıb öz məlumatlarınızı daxil edin."
    )

model = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))


def setup_database():
    """Nümunə DB yaradır"""
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
        INSERT OR IGNORE INTO orders (order_number, item, status, customer_email)
        VALUES ('12345', 'Şlanq', 'delivered', 'musteri@example.com')
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

    final_reply: Optional[str]
    messages: List[Dict[str, Any]]


# ============================================================
# NODES
# ============================================================
def read_query(state: CustomerRefundState):
    "Agent reads and logs incoming query"
    customer_message = state['customer_message']
    return {}


def classify_query(state: CustomerRefundState):
    "Agent uses an LLM to classify customer message"
    customer_message = state['customer_message']

    prompt = f"""
    As agent the butler, analyze this message and determine if it is related to refund or not.
    Answer clearly with either "refund" or "not related to refund".

    message: {customer_message}
    """
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()
    is_refund = "refund" in response_text and "not related to refund" not in response_text

    return {
        "is_refund_related": is_refund,
        "messages": state.get("messages", []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }


def normal_reply(state: CustomerRefundState):
    "answers for non-related queries"
    reply = "It is not related to refund. Sorry we can't help you"
    return {"final_reply": reply}


def ask_order_number(state: CustomerRefundState):
    "Ask for order number"
    order_number = input("Please tell us your order number: ")
    return {"order_number": order_number}


def check_order(state: CustomerRefundState):
    "check order in Database"
    order_number = state.get("order_number")
    order_data = None

    try:
        conn = sqlite3.connect("orders.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM orders WHERE order_number = ?",
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


def ask_again(state: CustomerRefundState):
    "Sifariş nömrəsi tapılmadıqda müştəridən yenidən soruşur"
    order_number = input("Order not found. Please enter a valid order number: ")
    return {"order_number": order_number}


def ask_reason(state: CustomerRefundState):
    "Ask reason from customer"
    reason = input("Please tell us the reason for the refund: ")
    return {"refund_reason": reason}


def classify_reason(state: CustomerRefundState):
    "classify reason using LLM"

    prompt = f"""Categorize refund reasons below:
    "wrong_item", "defect", "other"

    reason={state['refund_reason']}
    """
    response = model.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()

    if category not in ["wrong_item", "defect"]:
        category = "other"

    return {
        "reason_category": category,
        "messages": state.get("messages", []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }


def request_photo(state: CustomerRefundState):
    photo_info = input("Please describe/attach the photo of the item: ")
    return {"photo_requested": True, "additional_details": photo_info}


def ask_more_details(state: CustomerRefundState):
    details = input("Please tell us more detailed information: ")
    return {"additional_details": details}


def human_review(state: CustomerRefundState):
    """Refund sorğusunu real email ilə insan nəzərinə göndərir"""

    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    subject = f"New Refund Request - Order #{state.get('order_number', 'N/A')}"

    body = f"""
A new refund request needs human review.

Order number: {state.get('order_number')}
Order data: {state.get('order_data')}
Refund reason: {state.get('refund_reason')}
Reason category: {state.get('reason_category')}
Additional details: {state.get('additional_details')}
Photo requested: {state.get('photo_requested')}

Original customer message:
{state.get('customer_message')}
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Refund request for order {state.get('order_number')} sent to {receiver_email}")
    except Exception as e:
        print("Email sending failed:", e)

    return {}


def notify_customer(state: CustomerRefundState):
    reply = "Thank you! We will inform you within the next 7 days"
    return {"final_reply": reply}


# ============================================================
# LOGIC (routing functions)
# ============================================================
def route_by_intent(state: CustomerRefundState) -> str:
    """Mesajın refund ilə bağlı olub-olmadığına görə istiqaməti müəyyən edir"""
    if state["is_refund_related"]:
        return "refund_related"
    else:
        return "not_refund_related"


def route_by_order_validity(state: CustomerRefundState) -> str:
    """Sifariş nömrəsinin doğru olub-olmadığına görə istiqaməti müəyyən edir"""
    if state["order_valid"]:
        return "valid"
    else:
        return "invalid"


def route_by_reason(state: CustomerRefundState) -> str:
    """Refund səbəbinin kateqoriyasına görə istiqaməti müəyyən edir"""
    category = state.get("reason_category")

    if category in ["wrong_item", "defect"]:
        return "request_photo"
    else:
        return "ask_more_details"


# ============================================================
# EDGES (build the graph)
# ============================================================
builder = StateGraph(CustomerRefundState)

builder.add_node("read_query", read_query)
builder.add_node("classify_query", classify_query)
builder.add_node("normal_reply", normal_reply)
builder.add_node("ask_order_number", ask_order_number)
builder.add_node("check_order", check_order)
builder.add_node("ask_again", ask_again)
builder.add_node("ask_reason", ask_reason)
builder.add_node("classify_reason", classify_reason)
builder.add_node("request_photo", request_photo)
builder.add_node("ask_more_details", ask_more_details)
builder.add_node("human_review", human_review)
builder.add_node("notify_customer", notify_customer)

builder.add_edge(START, "read_query")
builder.add_edge("read_query", "classify_query")

builder.add_edge("normal_reply", END)

builder.add_edge("ask_order_number", "check_order")
builder.add_edge("ask_again", "check_order")
builder.add_edge("ask_reason", "classify_reason")

builder.add_edge("request_photo", "human_review")
builder.add_edge("ask_more_details", "human_review")

builder.add_edge("human_review", "notify_customer")
builder.add_edge("notify_customer", END)

builder.add_conditional_edges(
    "classify_query",
    route_by_intent,
    {
        "refund_related": "ask_order_number",
        "not_refund_related": "normal_reply"
    }
)

builder.add_conditional_edges(
    "check_order",
    route_by_order_validity,
    {
        "valid": "ask_reason",
        "invalid": "ask_again"
    }
)

builder.add_conditional_edges(
    "classify_reason",
    route_by_reason,
    {
        "request_photo": "request_photo",
        "ask_more_details": "ask_more_details"
    }
)

graph = builder.compile()


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    refund_message = {
        "customer_message": "I received the wrong item and want a refund.",
        "is_refund_related": None,
        "order_number": None,
        "order_valid": None,
        "order_data": None,
        "refund_reason": None,
        "reason_category": None,
        "photo_requested": False,
        "additional_details": None,
        "final_reply": None,
        "messages": []
    }

    result = graph.invoke(refund_message)
    print(result.get("final_reply"))
