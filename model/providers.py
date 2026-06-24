from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@dataclass
class ProviderConfig:
    provider: str
    model_name: str


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
    def invoke(self, prompt: str) -> str:
        return f"这是离线演示模型回复：{prompt}"

    def as_langchain_model(self):
        return MockChatModel()


class TongyiProvider:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def as_langchain_model(self):
        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=self.model_name)


def build_model_provider(config: ProviderConfig):
    if config.provider == "mock":
        return MockProvider()
    if config.provider == "tongyi":
        return TongyiProvider(config.model_name)
    raise ValueError(f"Unsupported model provider: {config.provider}")
