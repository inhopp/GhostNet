import os
import torch
import torch.nn as nn
from data import generate_loader
from option import get_option

@torch.no_grad()
def main(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    ft_path = os.path.join(opt.ckpt_root, opt.data_name, "best_epoch.pt")

    model = torch.hub.load('', 'ghostnet', opt.num_classes, source='local').to(dev)
    model.to(dev)
    if opt.multigpu:
        model = nn.DataParallel(model, device_ids=opt.device_ids).to(dev)
    model.load_state_dict(torch.load(ft_path))

    test_loader = generate_loader('test', opt)

    num_correct, num_total = 0, 0
    model.eval()
    for inputs in test_loader:
        images = inputs[0].to(dev)
        labels = inputs[1].to(dev)

        outputs = model(images)
        _, preds = torch.max(outputs.detach(), 1)
        num_correct += (preds == labels).sum().item()
        num_total += labels.size(0)

    print("Test Acc: {:.4f}".format(num_correct / num_total * 100))

if __name__ == '__main__':
    opt = get_option()
    main(opt)
