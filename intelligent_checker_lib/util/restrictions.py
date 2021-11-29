import json
import jsonschema
from jsonschema import validate
import yaml


class RestrictionHandler:
    current: dict = {}

    @staticmethod
    def __load_config(path: str) -> dict:
        try:
            with open(path, "r") as stream:
                document = yaml.safe_load(stream)
        except OSError as e:
            print("Read config error:", e)
            return {}
        except yaml.YAMLError as e:
            print("Yaml error:", e)
            return {}
        return document

    @staticmethod
    def __validate_config(obj: dict) -> bool:
        scheme_path = "restrictions_schema.json"
        try:
            with open(scheme_path, "r") as file:
                schema = json.load(file)
            validate(instance=obj, schema=schema)
        except OSError as e:
            print("Open file error:", e)
            return False
        except json.decoder.JSONDecodeError as e:
            print("Parse schema error:", e)
            return False
        except jsonschema.exceptions.ValidationError as err:
            print("Validation error:", err)
            return False
        return True

    @staticmethod
    def load(path: str):
        path_to_default_config = "default_config.yaml"
        custom_config = RestrictionHandler.__load_config(path)
        default_config = RestrictionHandler.__load_config(path_to_default_config)
        if RestrictionHandler.__validate_config(custom_config) and \
                RestrictionHandler.__validate_config(default_config):
            default_config.update(custom_config)
            RestrictionHandler.current = default_config
        else:
            RestrictionHandler.current = {}

    @staticmethod
    def has(key: str) -> bool:
        return any(restriction["name"] == key for restriction in RestrictionHandler.current["restrictions"])

    @staticmethod
    def get(key: str) -> tuple[float, float]:
        restriction = [restriction for restriction in RestrictionHandler.current["restrictions"] if restriction["name"] == key][0]
        return restriction["value"][0], restriction["value"][1]
