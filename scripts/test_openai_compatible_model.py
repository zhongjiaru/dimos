import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(x: int, y: int) -> int:
    """Add two integers together."""
    return x + y


def main() -> None:
    api_key = os.environ["OPENAI_COMPAT_API_KEY"]
    model = os.environ["OPENAI_COMPAT_MODEL"]
    base_url = os.environ["OPENAI_COMPAT_BASE_URL"]

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    chat = ChatOpenAI(**kwargs)

    print("=== Plain Text Test ===")
    response = chat.invoke(
        [HumanMessage(content="Translate this sentence from English to French. I love programming.")]
    )
    print(response.content)

    print("\n=== Tool Calling Test ===")
    tool_chat = chat.bind_tools([add])
    tool_response = tool_chat.invoke(
        [HumanMessage(content="What is 33333 + 100? Use the add tool.")],
    )
    print("content:", tool_response.content)
    print("tool_calls:", getattr(tool_response, "tool_calls", None))

    print("\n=== Full Agent Test ===")
    agent = create_agent(
        model=chat,
        tools=[add],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )
    agent_result = agent.invoke(
        {"messages": [HumanMessage(content="What is 33333 + 100? Use the add tool.")]},
    )

    messages = agent_result["messages"]
    final_message = messages[-1]
    print("final content:", final_message.content)
    print("all message types:", [type(message).__name__ for message in messages])


if __name__ == "__main__":
    main()
