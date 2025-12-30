
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies.selection import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination import DefaultTerminationStrategy
from semantic_kernel.functions import KernelFunction

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Semantic Kernel Multi-Agent Chatbot")

kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        service_id="openai",
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL
    )
)

user_chat_agent = ChatCompletionAgent(
    kernel=kernel,
    name="UserChatAgent",
    instructions="You chat with users and route tasks."
)

content_planner_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ContentPlannerAgent",
    instructions="You create structured content plans."
)

topics_generator_agent = ChatCompletionAgent(
    kernel=kernel,
    name="TopicsGeneratorAgent",
    instructions="You generate topic ideas."
)

agents = [user_chat_agent, content_planner_agent, topics_generator_agent]

selection_function = KernelFunction.from_prompt(
    """Choose agent:
- ContentPlannerAgent for planning
- TopicsGeneratorAgent for topics
- UserChatAgent otherwise
User input: {{$input}}"""
)

selection_strategy = KernelFunctionSelectionStrategy(
    kernel=kernel,
    function=selection_function,
    result_parser=lambda x: x.strip()
)

termination_strategy = DefaultTerminationStrategy(max_iterations=5)

group_chat = AgentGroupChat(
    agents=agents,
    selection_strategy=selection_strategy,
    termination_strategy=termination_strategy
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    group_chat.add_user_message(req.message)
    responses = []
    async for msg in group_chat.invoke():
        responses.append({"agent": msg.author, "content": msg.content})
    return {"conversation": responses}

@app.get("/")
def root():
    return {"status": "Running"}
