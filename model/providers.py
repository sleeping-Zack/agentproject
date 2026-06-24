"""模型 Provider 抽象：把不同厂商 / 自部署模型统一成同一接口。

为什么需要：实际生产里你会同时挂多家模型（豆包做主、通义做备、本地 vLLM
做长上下文兜底）。如果代码里直接 import ChatTongyi，换模型就要改全工程。
ProviderConfig + build_model_provider 把这件事收敛在一个文件里。

实现细节：豆包 / OpenAI-compatible / vLLM 都遵循 OpenAI Chat Completions
协议（火山方舟提供兼容端点），所以共享同一个 ChatOpenAI 客户端，只是
base_url 与默认 model_name 不同。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = messages[-1].content if messages else ""
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=f"这是离线演示模型回复：{content}"))
            ]
        )

    def bind_tools(self, tools, **kwargs):
        return self


class MockProvider:
    name = "mock"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def invoke(self, prompt: str) -> str:
        return f"这是离线演示模型回复：{prompt}"

    def as_langchain_model(self):
        return MockChatModel()


class TongyiProvider:
    name = "tongyi"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def as_langchain_model(self):
        from langchain_community.chat_models.tongyi import ChatTongyi
        return ChatTongyi(model=self.config.model_name)


class OpenAICompatibleProvider:
    """适配任何 OpenAI 兼容端点：豆包(火山方舟) / 自建 vLLM / Together / 其他。

    豆包：base_url=https://ark.cn-beijing.volces.com/api/v3, api_key_env=ARK_API_KEY,
          model_name=ep-2024xxxx-yyy（你的接入点 ID）
    vLLM：base_url=http://localhost:8000/v1, api_key_env=随便, model_name=本地权重名
    """
    name = "openai_compatible"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def as_langchain_model(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "使用 OpenAI 兼容 provider 需要安装 langchain-openai，"
                "请执行 pip install langchain-openai"
            ) from exc
        api_key_env = self.config.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env, "dummy-key")
        kwargs = {
            "model": self.config.model_name,
            "api_key": api_key,
            "base_url": self.config.base_url,
        }
        kwargs.update(self.config.extra or {})
        return ChatOpenAI(**kwargs)


class DoubaoProvider(OpenAICompatibleProvider):
    """火山方舟豆包模型。继承 OpenAI 兼容，仅默认 base_url 与 env 不同。"""
    name = "doubao"

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        if not config.api_key_env:
            config.api_key_env = "ARK_API_KEY"
        super().__init__(config)


class VLLMProvider(OpenAICompatibleProvider):
    """本地 vLLM / SGLang 等自部署推理服务。"""
    name = "vllm"

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not config.api_key_env:
            config.api_key_env = "VLLM_API_KEY"
        super().__init__(config)


_PROVIDER_REGISTRY = {
    "mock": MockProvider,
    "tongyi": TongyiProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "doubao": DoubaoProvider,
    "vllm": VLLMProvider,
}


def build_model_provider(config: ProviderConfig):
    cls = _PROVIDER_REGISTRY.get(config.provider)
    if cls is None:
        raise ValueError(f"Unsupported model provider: {config.provider}")
    return cls(config)
