from dataclasses import dataclass

import pytest

from model.providers import ProviderConfig, build_model_provider
from model.router import ModelRouter, ProviderEntry, NoAvailableModelError


def _mock_entry(name: str = "mock-1") -> ProviderEntry:
    return ProviderEntry(
        config=ProviderConfig(provider="mock", model_name=name),
        weight=10,
    )


def test_build_doubao_provider_sets_default_base_url():
    provider = build_model_provider(ProviderConfig(provider="doubao", model_name="ep-xxx"))
    # 不需要真实调用，只断言配置已被填充
    assert provider.config.base_url.startswith("https://ark")
    assert provider.config.api_key_env == "ARK_API_KEY"


def test_build_vllm_provider_defaults_to_localhost():
    provider = build_model_provider(ProviderConfig(provider="vllm", model_name="qwen-7b"))
    assert "/v1" in provider.config.base_url


def test_router_select_returns_highest_weight():
    router = ModelRouter()
    low = _mock_entry("low")
    low.weight = 1
    high = _mock_entry("high")
    high.weight = 10
    router.register(low)
    router.register(high)
    assert router.select().config.model_name == "high"


def test_router_invoke_falls_back_when_primary_fails():
    router = ModelRouter()
    primary = _mock_entry("primary")
    primary.weight = 10
    fallback = _mock_entry("fallback")
    fallback.weight = 1
    router.register(primary)
    router.register(fallback)

    attempts = []

    def fn(model):
        attempts.append(model)
        if len(attempts) == 1:
            raise RuntimeError("primary boom")
        return "ok"

    # 让 primary 直接失败到熔断阈值
    primary.breaker.failure_threshold = 1
    result = router.invoke(fn)
    assert result == "ok"
    assert len(attempts) == 2


def test_router_raises_when_no_provider_matches():
    router = ModelRouter()
    with pytest.raises(NoAvailableModelError):
        router.select(scene="long_context")


def test_router_filters_by_tenant():
    router = ModelRouter()
    tenant_only = _mock_entry("vip")
    tenant_only.tenants = ["vip-a"]
    router.register(tenant_only)
    public = _mock_entry("public")
    public.weight = 1
    router.register(public)

    chosen = router.select(tenant_id="vip-a")
    assert chosen.config.model_name == "vip"

    chosen = router.select(tenant_id="other")
    assert chosen.config.model_name == "public"
