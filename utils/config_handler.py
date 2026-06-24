"""
yaml
k: v
"""
import os

import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"))


def _split_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_agent_env_overrides(config: dict) -> dict:
    overrides = {
        "default_user_id": os.getenv("AGENT_DEFAULT_USER_ID"),
        "default_user_location": os.getenv("AGENT_DEFAULT_USER_LOCATION"),
        "current_month": os.getenv("AGENT_CURRENT_MONTH"),
    }
    for key, value in overrides.items():
        if value:
            config[key] = value

    allowed_tools = os.getenv("AGENT_ALLOWED_TOOLS")
    if allowed_tools:
        config["allowed_tools"] = _split_csv(allowed_tools)

    trace_enabled = os.getenv("AGENT_TRACE_ENABLED")
    if trace_enabled:
        config["trace_enabled"] = trace_enabled.lower() in {"1", "true", "yes", "on"}

    return config


def _apply_rag_env_overrides(config: dict) -> dict:
    model_provider = os.getenv("AGENT_MODEL_PROVIDER")
    if model_provider:
        config["model_provider"] = model_provider
    chat_model_name = os.getenv("AGENT_CHAT_MODEL_NAME")
    if chat_model_name:
        config["chat_model_name"] = chat_model_name
    embedding_model_name = os.getenv("AGENT_EMBEDDING_MODEL_NAME")
    if embedding_model_name:
        config["embedding_model_name"] = embedding_model_name
    return config

def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return _apply_rag_env_overrides(yaml.load(f, Loader=yaml.FullLoader))


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return _apply_agent_env_overrides(yaml.load(f, Loader=yaml.FullLoader))


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
