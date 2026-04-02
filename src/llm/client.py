"""Abstracted LLM client for model-agnostic AI interactions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    """LLM roles with different information visibility."""

    REFEREE = "referee"  # Full knowledge: all hidden info, full state, full rules
    RULES_INTERPRETER = "rules_interpreter"  # AI hidden + visible state + rules
    STRATEGIST = "strategist"  # AI hidden + visible state + legal moves + strategy


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    raw_response: Any
    model: str
    input_tokens: int
    output_tokens: int

    def as_json(self) -> dict:
        """Parse content as JSON, stripping markdown code fences if present."""
        import json

        text = self.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines[1:] if not l.strip() == "```"]
            text = "\n".join(lines)
        return json.loads(text)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: The user's message/query.
            role: Optional role hint for logging/debugging.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with the model's output.
        """
        pass

    @abstractmethod
    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: The user's message/query.
            images: List of image bytes (PNG, JPEG, etc.).
            role: Optional role hint for logging/debugging.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with the model's output.
        """
        pass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    display_name: str
    default_model: str
    available_models: list[str] = field(default_factory=list)
    requires_env: list[str] = field(default_factory=list)


# Provider configurations
PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        display_name="Anthropic (Direct API)",
        default_model="claude-sonnet-4-20250514",
        available_models=[
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-3-5-20241022",
        ],
        requires_env=["ANTHROPIC_API_KEY"],
    ),
    "bedrock": ProviderConfig(
        name="bedrock",
        display_name="AWS Bedrock (Claude)",
        default_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        available_models=[
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-haiku-4-20250514-v1:0",
        ],
        requires_env=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    ),
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        default_model="gpt-4o",
        available_models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"],
        requires_env=["OPENAI_API_KEY"],
    ),
    "ollama": ProviderConfig(
        name="ollama",
        display_name="Ollama (Local)",
        default_model="llama3.2",
        available_models=["llama3.2", "mistral", "mixtral", "qwen2.5", "deepseek-r1"],
        requires_env=[],  # No API key needed
    ),
}


def list_providers() -> list[ProviderConfig]:
    """List all available providers."""
    return list(PROVIDERS.values())


def get_provider(name: str) -> ProviderConfig | None:
    """Get a provider configuration by name."""
    return PROVIDERS.get(name)


class ClaudeClient(LLMClient):
    """Claude API client implementation (direct Anthropic API)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model identifier to use.
        """
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.provider = "anthropic"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion."""
        import base64

        # Build content with images and text
        content: list[dict] = []

        for image_bytes in images:
            # Detect media type from bytes
            media_type = self._detect_media_type(image_bytes)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )

        content.append({"type": "text", "text": user_prompt})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @staticmethod
    def _detect_media_type(image_bytes: bytes) -> str:
        """Detect image media type from magic bytes."""
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif image_bytes[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif image_bytes[:4] == b"GIF8":
            return "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        else:
            # Default to JPEG if unknown
            return "image/jpeg"


class BedrockClient(LLMClient):
    """Claude via AWS Bedrock client implementation."""

    def __init__(
        self,
        model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        region: str | None = None,
    ):
        """Initialize the Bedrock client.

        Args:
            model: Bedrock model identifier.
            region: AWS region. If None, uses default from environment.
        """
        import anthropic

        kwargs = {}
        if region:
            kwargs["aws_region"] = region
        self.client = anthropic.AsyncAnthropicBedrock(**kwargs)
        self.model = model
        self.provider = "bedrock"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion."""
        import base64

        content: list[dict] = []

        for image_bytes in images:
            media_type = ClaudeClient._detect_media_type(image_bytes)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )

        content.append({"type": "text", "text": user_prompt})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model identifier to use.
        """
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion."""
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion."""
        import base64

        content: list[dict] = []

        for image_bytes in images:
            media_type = ClaudeClient._detect_media_type(image_bytes)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    },
                }
            )

        content.append({"type": "text", "text": user_prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )


class OllamaClient(LLMClient):
    """Ollama local LLM client implementation."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ):
        """Initialize the Ollama client.

        Args:
            model: Model name to use.
            host: Ollama server URL.
        """
        self.model = model
        self.host = host
        self.provider = "ollama"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            raw_response=data,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion."""
        import base64
        import httpx

        # Ollama uses 'images' array with base64 encoded data
        image_data = [base64.b64encode(img).decode("utf-8") for img in images]

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt,
                            "images": image_data,
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            raw_response=data,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def create_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: The LLM provider ("anthropic", "bedrock", "openai", "ollama").
        api_key: API key for the provider (not needed for bedrock/ollama).
        model: Model identifier to use. If None, uses provider default.
        **kwargs: Additional provider-specific arguments.

    Returns:
        An LLMClient instance.

    Raises:
        ValueError: If provider is not supported.
    """
    config = get_provider(provider)
    if config is None:
        # Support legacy "claude" name
        if provider == "claude":
            provider = "anthropic"
            config = get_provider(provider)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    default_model = config.default_model if config else None

    if provider == "anthropic":
        return ClaudeClient(
            api_key=api_key,
            model=model or default_model or "claude-sonnet-4-20250514",
        )
    elif provider == "bedrock":
        return BedrockClient(
            model=model or default_model or "us.anthropic.claude-sonnet-4-20250514-v1:0",
            region=kwargs.get("region"),
        )
    elif provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model or default_model or "gpt-4o",
        )
    elif provider == "ollama":
        return OllamaClient(
            model=model or default_model or "llama3.2",
            host=kwargs.get("host", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
