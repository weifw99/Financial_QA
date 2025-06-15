
import os

from jinja2 import Template, meta
from ruamel.yaml import YAML

def render_template(config_path: str,  render_env: dict={}) -> str:
    """
    render the template based on the environment or render_env

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    with open(config_path, "r") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    context.update(render_env)
    print(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


def load_config(config_path="config.yaml"):
    """
    加载配置文件
    :param config_path:
    :return:
    """
    # dataset_cls=TSDatasetH step_len=20 num_timesteps=20 PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 qrun conf.yaml --experiment_name my_exp
    render_env = {"dataset_cls": 'TSDatasetH',
                  "num_timesteps": 20,
                  "step_len": 20}
    rendered_yaml = render_template(config_path, render_env=render_env)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)
    return config

