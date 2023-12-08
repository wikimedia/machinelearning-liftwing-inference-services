import yaml


def get_config(path: str = "python/config.yaml", key: str = None):
    with open(path) as yf:
        config = yaml.safe_load(yf)
    if key:
        return config[key]
    return config
