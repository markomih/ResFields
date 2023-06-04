datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import dycheck_parser, dysdf_dataset, dysurf_dataset
