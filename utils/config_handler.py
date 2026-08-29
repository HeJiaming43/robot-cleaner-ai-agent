import os
import yaml
from utils.path_tool import get_abs_path


def _load_yaml_config(config_path: str, encoding: str = "utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


rag_conf = _load_yaml_config(get_abs_path("config/rag.yml"))
chroma_conf = _load_yaml_config(get_abs_path("config/chroma.yml"))
prompts_conf = _load_yaml_config(get_abs_path("config/prompts.yml"))
agent_conf = _load_yaml_config(get_abs_path("config/agent.yml"))


if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
