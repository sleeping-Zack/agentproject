"""模型路由：按租户 / 场景 / 健康度选择 provider，并在主模型熔断时降级备用。

设计思路：
    - 多个 ProviderConfig 注册到 ModelRouter，每个标记一个 scene（默认/长上下文/低成本…）
    - select() 按上下文里的 tenant_id + scene 选主模型；不可用则退到 fallbacks
    - 健康度由 services.circuit_breaker 提供（成功/失败计数 + 半开探测）
    - 选中后通过 ContextVar 把 model 字段注入请求上下文，结构化日志自动带上

只在被实际 invoke 时才解析 langchain model，避免启动期就强依赖所有 SDK。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional

from model.providers import ProviderConfig, build_model_provider
from observability.context import bind_request_context, request_context
from services.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logger_handler import logger


DEFAULT_SCENE = "default"


@dataclass
class ProviderEntry:
    config: ProviderConfig
    scene: str = DEFAULT_SCENE
    tenants: List[str] = field(default_factory=list)  # 空表示对所有租户可用
    weight: int = 1
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    @property
    def name(self) -> str:
        return f"{self.config.provider}:{self.config.model_name}"


class NoAvailableModelError(RuntimeError):
    pass


class ModelRouter:
    def __init__(self) -> None:
        self._entries: List[ProviderEntry] = []
        self._lock = RLock()

    def register(self, entry: ProviderEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def list_entries(self) -> List[ProviderEntry]:
        with self._lock:
            return list(self._entries)

    def _candidates(self, scene: str, tenant_id: Optional[str]) -> List[ProviderEntry]:
        out: List[ProviderEntry] = []
        for entry in self._entries:
            if entry.scene != scene and entry.scene != DEFAULT_SCENE:
                continue
            if entry.tenants and tenant_id and tenant_id not in entry.tenants:
                continue
            out.append(entry)
        out.sort(key=lambda e: (0 if e.scene == scene else 1, -e.weight))
        return out

    def select(self, scene: str = DEFAULT_SCENE,
               tenant_id: Optional[str] = None) -> ProviderEntry:
        candidates = self._candidates(scene, tenant_id)
        if not candidates:
            raise NoAvailableModelError(
                f"no provider registered for scene={scene} tenant={tenant_id}"
            )
        healthy = [c for c in candidates if c.breaker.allow()]
        chosen = healthy[0] if healthy else candidates[0]
        return chosen

    def invoke(self, fn, scene: str = DEFAULT_SCENE, tenant_id: Optional[str] = None):
        """执行一次模型调用，失败按 fallbacks 顺序降级。fn 接收 langchain_model。"""
        tenant_id = tenant_id or request_context().tenant_id
        candidates = self._candidates(scene, tenant_id)
        if not candidates:
            raise NoAvailableModelError(
                f"no provider registered for scene={scene} tenant={tenant_id}"
            )
        last_exc: Optional[BaseException] = None
        for entry in candidates:
            if not entry.breaker.allow():
                logger.warning("[router]熔断打开，跳过", extra={"provider": entry.name})
                continue
            try:
                with bind_request_context(model=entry.name):
                    model = build_model_provider(entry.config).as_langchain_model()
                    result = fn(model)
                entry.breaker.record_success()
                return result
            except CircuitOpenError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                entry.breaker.record_failure()
                last_exc = exc
                logger.warning(
                    "[router]调用失败，尝试降级",
                    extra={"provider": entry.name, "error": str(exc)},
                )
                continue
        raise NoAvailableModelError(
            f"all providers exhausted for scene={scene}: {last_exc}"
        )

    def health(self) -> Dict[str, Dict]:
        return {e.name: {"scene": e.scene, "state": e.breaker.state.value,
                         "failures": e.breaker.failure_count}
                for e in self._entries}


def build_default_router_from_config(rag_conf: Dict) -> ModelRouter:
    """从 rag.yml 主配置构造一个最小 router；额外 provider 可由调用方继续 register。"""
    router = ModelRouter()
    router.register(ProviderEntry(
        config=ProviderConfig(
            provider=os.getenv("MODEL_PROVIDER", rag_conf.get("model_provider", "tongyi")),
            model_name=os.getenv("CHAT_MODEL_NAME", rag_conf["chat_model_name"]),
        ),
        scene=DEFAULT_SCENE,
        weight=10,
    ))
    fallback = rag_conf.get("fallback_provider")
    if fallback:
        router.register(ProviderEntry(
            config=ProviderConfig(
                provider=fallback,
                model_name=rag_conf.get("fallback_model_name",
                                        rag_conf["chat_model_name"]),
                base_url=rag_conf.get("fallback_base_url"),
                api_key_env=rag_conf.get("fallback_api_key_env"),
            ),
            scene=DEFAULT_SCENE,
            weight=1,
        ))
    return router
