"""
Food Delivery Order-Tracking Assistant
Uses modern LangChain (RunnableWithMessageHistory) instead of the
deprecated ConversationChain/ConversationBufferMemory classes,
to avoid ModuleNotFoundError with newer langchain versions.
"""

import re
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Order data ----------------
orders = {
    "#101": "Preparing",
    "#102": "Out for delivery",
    "#103": "Delivered"
}

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="google_api_key"   # replace with your real Gemini API key
)

# ---------------- Prompt with memory placeholder ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a polite food delivery order-tracking assistant. "
               "Use the conversation history to answer follow-up questions naturally "
               "(e.g. if the user previously asked about an order and now asks 'what about #102?', "
               "understand they mean order #102)."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- In-memory chat history store ----------------
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user-session-1"}}

# ---------------- Main loop ----------------
queried_orders = set()

print("Food Delivery Assistant")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    match = re.search(r"#\d+", user_input)

    if match:
        order_id = match.group()
        queried_orders.add(order_id)

        if order_id in orders:
            user_prompt = (
                f"The customer asked for order {order_id}. "
                f"The current order status is '{orders[order_id]}'. "
                f"Reply politely with the status."
            )
        else:
            user_prompt = (
                f"Order {order_id} does not exist. "
                f"Tell the customer politely that the order ID was not found."
            )
    else:
        # Let memory + LLM handle follow-up questions naturally
        user_prompt = user_input

    response = conversation.invoke(
        {"input": user_prompt},
        config=config
    )

    print("Assistant:", response.content)

print("\nSession Summary")
print("Unique Orders Queried:", len(queried_orders))