from model import get_ghostnet

def ghostnet(num_classes=1000, **kwargs):
    return get_ghostnet(num_classes, **kwargs)