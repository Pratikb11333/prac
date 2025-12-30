
import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import (
    ChatCompletionAgent,
    GroupChatOrchestration
)
from semantic_kernel.agents.orchestration import (
    RoundRobinGroupChatManager
)

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="SK One-on-One Multi-Turn Chatbot")

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
Coordinate with other agents internally if needed,
but always produce the final answer for the user.
"""
)

content_planner_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ContentPlannerAgent",
    instructions="""
You create structured content plans when required.
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
# GROUP CHAT MANAGER
# --------------------------------------------------
manager = RoundRobinGroupChatManager(max_rounds=6)

# --------------------------------------------------
# MULTI-TURN CONVERSATION STORE
# --------------------------------------------------
conversations = {}

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

    # Restore or create orchestration
    if request.conversation_id and request.conversation_id in conversations:
        orchestration = conversations[request.conversation_id]
        conversation_id = request.conversation_id
    else:
        conversation_id = str(uuid.uuid4())
        orchestration = GroupChatOrchestration(
            members=agents,
            manager=manager
        )
        conversations[conversation_id] = orchestration

    # Run orchestration
    await orchestration.invoke(task=request.message)

    # Get last assistant message only
    last_assistant_message = None
    for msg in reversed(orchestration.chat_history.messages):
        if msg.role == "assistant":
            last_assistant_message = msg
            break

    return {
        "conversation_id": conversation_id,
        "agent": last_assistant_message.author_name,
        "response": last_assistant_message.content
    }

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "One-on-One Multi-Turn Bot Running ✅"}
