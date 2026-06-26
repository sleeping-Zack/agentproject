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
