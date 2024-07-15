import json
import os


class JSONFileManager:
    instances: list['JSONFileManager'] = []

    def __init__(self, path: str, default: dict = None) -> None:
        """
        Constructor of a JSON file manager.
        :param path: file path of the json file
        :param default: default data for the json file
        """
        self.path = path
        self.default = {} if default is None else default

        self.data = {}
        self.reload()

        JSONFileManager.instances.append(self)

    def restore_default(self) -> None:
        """
        Restore data dictionary to default dictionary.
        :return:
        """
        self.data = self.default.copy()

    def save(self, reload: bool = True) -> None:
        """
        Save data in dictionary to the file.
        :param reload: reload other JSON file manager with the same path
        :return:
        """
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path))

        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=4)

        if reload:
            for instance in self.instances:
                if instance.path == self.path:
                    instance.reload()

    def reload(self) -> None:
        """
        Reload data from file.
        :return:
        """
        try:
            with open(self.path, "r") as f:
                self.data = json.load(f)
        except:
            self.restore_default()