import functools
from typing import Collection
from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openai.resources.chat.completions.completions import (
    Completions as _ChatCompletions,
    AsyncCompletions as _AsyncChatCompletions,
)
from openai.resources.completions import (
    Completions as _Completions,
    AsyncCompletions as _AsyncCompletions,
)
from openai.resources.embeddings import (
    Embeddings as _Embeddings,
    AsyncEmbeddings as _AsyncEmbeddings,
)
import json

tracer = trace.get_tracer("llamatry")


class OpenAIInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        self._original_chatcompletion_create = _ChatCompletions.create
        self._original_chatcompletion_acreate = _AsyncChatCompletions.create
        self._original_create = _Completions.create
        self._original_acreate = _AsyncCompletions.create
        self._original_embeddings_create = _Embeddings.create
        self._original_embeddings_acreate = _AsyncEmbeddings.create

        _ChatCompletions.create = self._trace_create(
            self._original_chatcompletion_create
        )
        _AsyncChatCompletions.create = self._trace_acreate(
            self._original_chatcompletion_acreate
        )
        _Completions.create = self._trace_create(self._original_create)
        _AsyncCompletions.create = self._trace_acreate(self._original_acreate)
        _Embeddings.create = self._trace_create(self._original_embeddings_create)
        _AsyncEmbeddings.create = self._trace_acreate(
            self._original_embeddings_acreate
        )

    def _uninstrument(self, **kwargs):
        _ChatCompletions.create = self._original_chatcompletion_create
        _AsyncChatCompletions.create = self._original_chatcompletion_acreate
        _Completions.create = self._original_completion_create
        _AsyncCompletions.create = self._original_completion_acreate
        _Embeddings.create = self._original_embeddings_create
        _AsyncEmbeddings.create = self._original_embeddings_acreate

    def instrumentation_dependencies(self) -> Collection[BaseInstrumentor]:
        return []

    @staticmethod
    def _handle_attributes(span, kwargs, response):
        """
        Set attributes on the span for the prompt and response, purposefully
        ignoring the prompt and response attributes as they are may be too large

        :param span: the span to set attributes on
        :param kwargs: the kwargs passed to the create method
        :param response: the response from the create method
        """
        for key, value in kwargs.items():
            if isinstance(value, (str, bool, float, int)):
                span.set_attribute(f"openai.create.{key}", value)
            elif isinstance(value, (list, dict)):
                span.set_attribute(f"openai.create.{key}", json.dumps(value))

        for key in ["id", "created", "model"]:
            value = getattr(response, key, None)
            if value is not None:
                span.set_attribute(f"openai.response.{key}", value)

        usage = getattr(response, "usage", None)
        if usage:
            for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
                value = getattr(usage, key, None)
                if value is not None:
                    span.set_attribute(f"openai.usage.{key}", value)
        span.set_attribute("openai.response", response.model_dump_json())

    def _trace_create(self, original_create):
        @functools.wraps(original_create)
        def wrapper(*args, **kwargs):
            stream = kwargs.get("stream", False)

            with tracer.start_as_current_span(
                f"openai.{original_create.__qualname__}",
            ) as span:
                if stream:

                    def stream_wrapper():
                        first_chunk = True
                        for resp in original_create(*args, **kwargs):
                            if first_chunk:
                                self._handle_attributes(span, kwargs, resp)
                                first_chunk = False
                            yield resp

                    return stream_wrapper()
                else:
                    response = original_create(*args, **kwargs)
                    self._handle_attributes(span, kwargs, response)
                    return response

        return wrapper

    def _trace_acreate(self, original_acreate):
        @functools.wraps(original_acreate)
        async def wrapper(*args, **kwargs):
            stream = kwargs.get("stream", False)
            with tracer.start_as_current_span(
                f"openai.{original_acreate.__qualname__}"
            ) as span:
                response = await original_acreate(*args, **kwargs)
                if stream:

                    async def stream_wrapper():
                        first_chunk = True
                        async for resp in response:
                            if first_chunk:
                                self._handle_attributes(span, kwargs, resp)
                                first_chunk = False
                            yield resp

                    return stream_wrapper()
                else:
                    self._handle_attributes(span, kwargs, response)
                    return response  # type: ignore

        return wrapper
