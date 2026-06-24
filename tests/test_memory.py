from agent.memory import ConversationMemory, InMemorySessionStore


def test_conversation_memory_keeps_bounded_history():
    memory = ConversationMemory(max_messages=3)

    memory.add_message("s1", "user", "第一条")
    memory.add_message("s1", "assistant", "第二条")
    memory.add_message("s1", "user", "第三条")
    memory.add_message("s1", "assistant", "第四条")

    assert memory.get_messages("s1") == [
        {"role": "assistant", "content": "第二条"},
        {"role": "user", "content": "第三条"},
        {"role": "assistant", "content": "第四条"},
    ]


def test_conversation_memory_tracks_profile_and_tool_results():
    memory = ConversationMemory()

    memory.update_profile("s1", {"user_id": "1001", "city": "深圳"})
    memory.set_last_tool_result("s1", "get_weather", "晴天")

    snapshot = memory.snapshot("s1")

    assert snapshot["profile"] == {"user_id": "1001", "city": "深圳"}
    assert snapshot["last_tool_results"] == {"get_weather": "晴天"}


def test_conversation_memory_persists_to_external_store():
    store = InMemorySessionStore()
    memory = ConversationMemory(max_messages=5, store=store)

    memory.add_message("s1", "user", "你好")
    memory.add_message("s1", "assistant", "你好，请问")

    fresh_memory = ConversationMemory(max_messages=5, store=store)
    assert fresh_memory.get_messages("s1") == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，请问"},
    ]


def test_conversation_memory_compresses_with_summarizer():
    store = InMemorySessionStore()

    def summarizer(messages, previous_summary):
        return f"已压缩{len(messages)}条对话，前情：{previous_summary or '无'}"

    memory = ConversationMemory(
        max_messages=10,
        store=store,
        summarizer=summarizer,
        summary_trigger=4,
        summary_keep_recent=2,
    )

    for i in range(5):
        memory.add_message("s1", "user", f"问题{i}")
        memory.add_message("s1", "assistant", f"回答{i}")

    messages = memory.get_messages("s1")
    assert messages[0]["role"] == "system"
    assert "已压缩" in messages[0]["content"]
    assert messages[-1]["role"] == "assistant"
    # 触发摘要后窗口必然短于全量 10 条（含 summary 头）
    assert 1 < len(messages) < 10


def test_conversation_memory_isolates_tenants():
    store = InMemorySessionStore()
    memory = ConversationMemory(max_messages=5, store=store)

    memory.add_message("s1", "user", "tenant a 的消息", tenant_id="tenant-a")
    memory.add_message("s1", "user", "tenant b 的消息", tenant_id="tenant-b")

    a = memory.get_messages("s1", tenant_id="tenant-a")
    b = memory.get_messages("s1", tenant_id="tenant-b")
    assert len(a) == 1 and a[0]["content"] == "tenant a 的消息"
    assert len(b) == 1 and b[0]["content"] == "tenant b 的消息"
