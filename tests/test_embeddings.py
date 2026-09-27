import os
import openai
import pytest
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from llamatry import OpenAIInstrumentor

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Set up OpenTelemetry
trace.set_tracer_provider(TracerProvider())

# Set up OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Instrument the OpenAI API
OpenAIInstrumentor().instrument()


def test_embed_create():
    response = openai.embeddings.create(
        input="Your text string goes here", model="text-embedding-ada-002"
    )
    assert response is not None


@pytest.mark.asyncio
async def test_embed_acreate():
    client = openai.AsyncOpenAI()
    response = await client.embeddings.create(
        input="Your text string goes here", model="text-embedding-ada-002"
    )
    assert response is not None
