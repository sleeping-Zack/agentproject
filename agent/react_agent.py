from langchain.agents import create_agent
from typing import Optional
from uuid import uuid4

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.memory import ConversationMemory
from observability.tracing import trace_recorder
from safety.security import UnsafeInputError, assert_safe_user_input


class ReactAgent:
    def __init__(self):
        self.memory = ConversationMemory()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, session_id: str = "default", request_id: Optional[str] = None):
        request_id = request_id or str(uuid4())
        trace_recorder.start_trace(request_id=request_id, session_id=session_id)
        try:
            assert_safe_user_input(query)
        except UnsafeInputError as e:
            yield f"请求未执行：{str(e)}\n"
            return

        history = self.memory.get_messages(session_id)
        input_dict = {
            "messages": history + [{"role": "user", "content": query}]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        latest_response = ""
        with trace_recorder.span(
                request_id,
                category="agent",
                name="execute_stream",
                metadata={"query": query, "history_count": len(history)},
        ):
            for chunk in self.agent.stream(
                    input_dict,
                    stream_mode="values",
                    context={"report": False, "request_id": request_id, "session_id": session_id},
            ):
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    latest_response = latest_message.content.strip() + "\n"
                    yield latest_response

        if latest_response:
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", latest_response.strip())


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
