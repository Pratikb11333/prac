import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from semantic_kernel.agents import (
    ChatCompletionAgent,
    AgentGroupChat
)

from semantic_kernel.agents.strategies.selection import (
    KernelFunctionSelectionStrategy
)

from semantic_kernel.agents.strategies.termination import (
    DefaultTerminationStrategy
)

from semantic_kernel.functions import KernelFunction

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="SK AgentGroupChat Multi-Turn Chatbot")

# --------------------------------------------------
# KERNEL
# --------------------------------------------------
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        service_id="openai",
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL
    )
)

# --------------------------------------------------
# AGENTS
# --------------------------------------------------
user_agent = ChatCompletionAgent(
    kernel=kernel,
    name="UserChatAgent",
    instructions="""
You are the main user-facing assistant.
Handle normal conversation and follow-ups.
"""
)

content_planner_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ContentPlannerAgent",
    instructions="""
You are a content strategist.
Create structured content plans when requested.
"""
)

topics_generator_agent = ChatCompletionAgent(
    kernel=kernel,
    name="TopicsGeneratorAgent",
    instructions="""
You generate topic ideas and brainstorming lists.
"""
)

agents = [
    user_agent,
    content_planner_agent,
    topics_generator_agent
]

# --------------------------------------------------
# AGENT SELECTION (SMART ROUTING)
# --------------------------------------------------
selection_function = KernelFunction.from_prompt(
    """
Choose the best agent for the user request.

Rules:
- Content planning / strategy → ContentPlannerAgent
- Topic ideas / brainstorming → TopicsGeneratorAgent
- Otherwise → UserChatAgent

Respond with ONLY the agent name.

User input:
{{$input}}
"""
)

selection_strategy = KernelFunctionSelectionStrategy(
    kernel=kernel,
    function=selection_function,
    result_parser=lambda r: r.strip()
)

termination_strategy = DefaultTerminationStrategy(
    max_iterations=5
)

# --------------------------------------------------
# MULTI-TURN CONVERSATION STORE
# --------------------------------------------------
conversations: dict[str, AgentGroupChat] = {}

def create_group_chat() -> AgentGroupChat:
    return AgentGroupChat(
        agents=agents,
        selection_strategy=selection_strategy,
        termination_strategy=termination_strategy
    )

# --------------------------------------------------
# API MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

# --------------------------------------------------
# CHAT ENDPOINT (ONE-ON-ONE RESPONSE)
# --------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):

    # Restore or create conversation
    if request.conversation_id and request.conversation_id in conversations:
        group_chat = conversations[request.conversation_id]
        conversation_id = request.conversation_id
    else:
        conversation_id = str(uuid.uuid4())
        group_chat = create_group_chat()
        conversations[conversation_id] = group_chat

    # Add user message
    group_chat.add_user_message(request.message)

    # Run agents
    last_message = None
    async for message in group_chat.invoke():
        last_message = message

    return {
        "conversation_id": conversation_id,
        "agent": last_message.author,
        "response": last_message.content
    }

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "AgentGroupChat bot running ✅"}
