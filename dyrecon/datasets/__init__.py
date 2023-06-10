datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import dysdf_dataset, video_dataset, tsdf_dataset
