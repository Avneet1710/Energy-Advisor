"""
EcoHome Energy Advisor agent.

The agent is an explicit LangGraph graph:

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

  Schema : AgentState, a message list that new messages are appended to
  Nodes  : "agent" (the LLM decides what to do) and "tools" (runs the requested tools)
  Edges  : START -> agent, agent -> tools or END (conditional), tools -> agent

Contract:
  Agent(instructions, model)       -> constructor with at least these two arguments
  Agent.invoke(question, context)  -> asks a question and returns the graph result, {"messages": [...]}
"""
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOL_KIT

load_dotenv()


# ---------------------------------------------------------------------------
# Schema: the state that flows through the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # add_messages appends new messages to the list instead of replacing it,
    # so every node sees the whole conversation, including tool calls and tool results.
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o-mini", temperature: float = 0.0, tools: Optional[list] = None):
        """
        Args:
            instructions (str): The system prompt that defines the Energy Advisor
            model (str): Chat model name, used through the Vocareum endpoint
            temperature (float): Sampling temperature (0.0 keeps evaluation runs stable)
            tools (list): Tools the agent may call (defaults to TOOL_KIT)
        """
        self.instructions = instructions
        self.tools = tools if tools is not None else TOOL_KIT

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )
        # The same model, but it can now ask for tool calls
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    # -----------------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------------
    def _agent_node(self, state: AgentState) -> dict:
        """LLM node: read the conversation, then either answer or request tool calls."""
        # The instructions are added on every call and are not stored in the state
        prompt = [SystemMessage(content=self.instructions)] + list(state["messages"])
        response = self.llm_with_tools.invoke(prompt)
        return {"messages": [response]}          # add_messages appends this to the conversation

    # -----------------------------------------------------------------------
    # Graph: nodes and edges
    # -----------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))   # a tool error becomes a message, not a crash

        # Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,                      # "tools" if the last message has tool calls, otherwise END
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")       # after the tools run, the LLM reads the results

        return workflow.compile()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def invoke(self, question: str, context: Optional[str] = None, config: Optional[dict] = None) -> dict:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question about energy optimization
            context (str): Optional extra context (for example the home's location or devices),
                added as a system message before the question
            config (dict): Optional LangGraph run config, for example {"recursion_limit": 40}

        Returns:
            dict: The graph result. result["messages"] holds the whole conversation, including tool calls
            and tool results. The advisor's final answer is result["messages"][-1].content.
        """
        messages = []
        if context:
            # Add some context to the question as a system message
            messages.append(("system", context))
        messages.append(("user", question))

        # Get response from the agent
        return self.graph.invoke({"messages": messages}, config=config)

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in self.tools]
