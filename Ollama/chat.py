from langchain_community.chat_models import ChatOllama
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationChain
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.schema import SystemMessage, HumanMessage

# Initialize chat model with optimized settings
chat_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.6,  # Balanced creativity
    num_predict=300   # Concise responses
)

# Set up memory to retain the last 10 exchanges
memory = ConversationBufferWindowMemory(
    k=10,
    return_messages=True,
    memory_key="chat_history"
)

# Create a conversation prompt
conversation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful business assistant. 
Provide clear, professional responses and remember context from our conversation.
Keep responses concise but informative."""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

# Build the conversation chain
conversation = ConversationChain(
    llm=chat_llm,
    memory=memory,
    prompt=conversation_prompt,
    verbose=False
)

# Example conversation flow
def chat_session():
    print("Business Assistant: Hello! How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Business Assistant: Goodbye!")
            break

        response = conversation.predict(input=user_input)
        print(f"Business Assistant: {response}")

# Example automated responses
responses = [
    "What's our current project status?",
    "Can you remind me about the client meeting details?",
    "What were the action items from our last discussion?"
]

for question in responses:
    answer = conversation.predict(input=question)
    print(f"Q: {question}")
    print(f"A: {answer}")