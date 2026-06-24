from agent.memory import ConversationMemory


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
