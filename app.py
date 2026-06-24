import time
from uuid import uuid4

import streamlit as st
from agent.react_agent import ReactAgent
from utils.streaming import get_final_response


def main() -> None:
    # 标题
    st.title("智扫通机器人智能客服")
    st.divider()

    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()

    if "message" not in st.session_state:
        st.session_state["message"] = []

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())

    for message in st.session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])

    # 用户输入提示词
    prompt = st.chat_input()

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})

        response_messages = []
        request_id = str(uuid4())
        with st.spinner("智能客服思考中..."):
            res_stream = st.session_state["agent"].execute_stream(
                prompt,
                session_id=st.session_state["session_id"],
                request_id=request_id,
            )

            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)

                    for char in chunk:
                        time.sleep(0.01)
                        yield char

            st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
            st.session_state["message"].append(
                {"role": "assistant", "content": get_final_response(response_messages)}
            )
            st.rerun()


if __name__ == "__main__":
    main()
