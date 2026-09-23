import sys
sys.path.insert(0, "backend")
from services.language import is_greeting

test_queries = [
    "Hello",
    "Hi",
    "Namaste",
    "Hello, what is the interest rate for personal loan?",
    "Hi, tell me about personal loan",
    "What is the CIBIL score requirement?",
    "Hello, can you help me with loan eligibility?",
    "Can you tell me about personal loans?",
    "Tell me about HDFC bank loan",
]

for q in test_queries:
    print(f"'{q}' -> is_greeting: {is_greeting(q)}")
