import json


class Config(dict):
    """ Persistable dictionary (JSON) to store experiment configs. """
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filename = filename

    def save(self):
        with open(self._filename, 'w') as f:
            data = {k: v for k, v in self.items()}
            data['_filename'] = self._filename
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        if '_filename' in data:
            filename = data.pop('_filename')
        return cls(filename, **data)
