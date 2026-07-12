from model.providers import MockProvider, ProviderConfig, TongyiProvider, build_model_provider


def test_mock_provider_returns_deterministic_answer():
    provider = MockProvider(ProviderConfig(provider="mock", model_name="offline"))

    assert provider.invoke("你好") == "这是离线演示模型回复：你好"


def test_build_provider_uses_mock_for_offline_demo():
    provider = build_model_provider(ProviderConfig(provider="mock", model_name="offline"))

    assert isinstance(provider, MockProvider)


def test_tongyi_provider_forwards_generation_parameters(monkeypatch):
    captured = {}

    class FakeChatTongyi:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "langchain_community.chat_models.tongyi.ChatTongyi",
        FakeChatTongyi,
    )
    provider = TongyiProvider(
        ProviderConfig("tongyi", "qwen-test", extra={"temperature": 0.0})
    )

    provider.as_langchain_model()

    assert captured == {"model": "qwen-test", "temperature": 0.0}
