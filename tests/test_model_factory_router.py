from model import factory


def test_chat_model_factory_uses_model_router(monkeypatch):
    calls = []

    class FakeRouter:
        def invoke(self, fn, scene="default", tenant_id=None):
            calls.append({"scene": scene, "tenant_id": tenant_id})
            return fn("fake-chat-model")

    monkeypatch.setattr(factory, "model_router", FakeRouter())

    model = factory.ChatModelFactory().generator(scene="report", tenant_id="tenant-a")

    assert model == "fake-chat-model"
    assert calls == [{"scene": "report", "tenant_id": "tenant-a"}]


def test_lazy_model_callable_falls_back_to_invoke_for_langchain_pipeline():
    calls = []

    class InvokeOnlyModel:
        def invoke(self, value, **kwargs):
            calls.append((value, kwargs))
            return "answer"

    lazy = factory._LazyModel(lambda: InvokeOnlyModel())

    assert lazy("prompt", config={"tag": "rag"}) == "answer"
    assert calls == [("prompt", {"config": {"tag": "rag"}})]


def test_lazy_model_prefers_runnable_invoke_over_legacy_callable():
    class RunnableModel:
        def __call__(self, _value):
            return "legacy-call"

        def invoke(self, _value):
            return "runnable-invoke"

    lazy = factory._LazyModel(lambda: RunnableModel())

    assert lazy("prompt") == "runnable-invoke"
    assert isinstance(lazy.resolve(), RunnableModel)
