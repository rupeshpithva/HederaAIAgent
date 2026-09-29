# main.py
import asyncio
import os
import re

from dotenv import load_dotenv
from hedera_agent_kit.langchain.toolkit import HederaLangchainToolkit
from hedera_agent_kit.plugins import (
    core_account_plugin,
    core_account_query_plugin,
    core_token_plugin,
    core_consensus_plugin,
)
from hedera_agent_kit.shared.configuration import Configuration, Context, AgentMode
from hiero_sdk_python import Client, Network, AccountId, PrivateKey
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

REQUIRED_ENV_VARS = ("HEDERA_ACCOUNT_ID", "HEDERA_PRIVATE_KEY", "OPENAI_API_KEY")

HEDERA_ACCOUNT_ID_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
OPENAI_API_KEY_PATTERN = re.compile(r"^sk-[A-Za-z0-9_-]{20,}$")


def check_env_vars():
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. Please set them in your .env file."
        )

    hedera_account_id = os.getenv("HEDERA_ACCOUNT_ID")
    if not HEDERA_ACCOUNT_ID_PATTERN.match(hedera_account_id):
        raise EnvironmentError(
            f"Invalid HEDERA_ACCOUNT_ID format: '{hedera_account_id}'. "
            "Expected shard.realm.num format, e.g. '0.0.123456'."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY_PATTERN.match(openai_api_key):
        raise EnvironmentError(
            "Invalid OPENAI_API_KEY format. Expected a key starting with 'sk-' "
            "followed by at least 20 characters."
        )


async def main():
    check_env_vars()

    # Hedera client setup (Testnet by default)
    account_id = AccountId.from_string(os.getenv("HEDERA_ACCOUNT_ID"))
    private_key = PrivateKey.from_string(os.getenv("HEDERA_PRIVATE_KEY"))
    client = Client(Network(network="testnet"))
    client.set_operator(account_id, private_key)

    # Prepare Hedera toolkit
    hedera_toolkit = HederaLangchainToolkit(
        client=client,
        configuration=Configuration(
            tools=[],  # Empty = load all tools from plugins
            plugins=[
                core_account_plugin,
                core_account_query_plugin,
                core_token_plugin,
                core_consensus_plugin,
            ],
            context=Context(
                mode=AgentMode.AUTONOMOUS,
                account_id=str(account_id),
            ),
        ),
    )

    tools = hedera_toolkit.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=MemorySaver(),
        system_prompt="You are a helpful assistant with access to Hedera blockchain tools and plugin tools",
    )

    print("Sending a message to the agent...")

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's Hedera Agents SDK and how to use it ?"}]},
        config={"configurable": {"thread_id": "1"}},
    )

    final_message_content = response["messages"][-1].content
    print("\n--- Agent Response ---")
    print(final_message_content)
    print("----------------------")


if __name__ == "__main__":
    asyncio.run(main())