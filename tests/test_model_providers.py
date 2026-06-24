from model.providers import MockProvider, ProviderConfig, build_model_provider


def test_mock_provider_returns_deterministic_answer():
    provider = MockProvider(ProviderConfig(provider="mock", model_name="offline"))

    assert provider.invoke("你好") == "这是离线演示模型回复：你好"


def test_build_provider_uses_mock_for_offline_demo():
    provider = build_model_provider(ProviderConfig(provider="mock", model_name="offline"))

    assert isinstance(provider, MockProvider)
