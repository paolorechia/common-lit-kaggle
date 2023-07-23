class Schema:
    @classmethod
    def to_dict(cls):
        data = {}
        for attr_name in dir(cls):
            # Hopefully ignore all private fields
            if not attr_name.startswith("_"):
                data[attr_name] = getattr(cls, attr_name)
        return data
