"""
LLM module for Ralph.

This module provides functions to create and configure LangChain chains
for interacting with Large Language Models.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ralph.config import RalphConfig

def get_chain(config: RalphConfig):
    """
    Create and return a simple LangChain chain for answering questions.

    This chain uses the configured Google GenAI model.

    Args:
        config (RalphConfig): The Ralph configuration object.

    Returns:
        Runnable: A LangChain runnable chain that accepts a dictionary with a "question" key
                  and returns a string response.

    Raises:
        ValueError: If the Google API key is not set in the configuration.
    """
    if not config.aiclient.google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Initialize the Google Gemini model
    # Using gemini-pro as a safe default, or user could specify via env vars if we expanded settings
    llm = ChatGoogleGenerativeAI(
        model=config.aiclient.model or "gemini-pro",
        google_api_key=config.aiclient.google_api_key.get_secret_value()
    )

    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ])

    # Create the chain
    chain = prompt | llm | StrOutputParser()

    return chain
