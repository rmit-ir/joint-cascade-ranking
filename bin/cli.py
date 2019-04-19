class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    @classmethod
    def from_parseargs(cls, args):
        config = cls()
        config.merge_parseargs(args)
        return config

    @classmethod
    def from_attr_dict(cls, other):
        config = cls()
        for key, val in other.items():
            setattr(config, key, val)
        return config

    def merge_parseargs(self, args):
        for key, val in vars(args).items():
            # this class is a `dict` so use `in` instead of `hasattr`
            if key in self.keys() and val is not None:
                setattr(self, key, val)
