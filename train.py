import os
from tqdm import tqdm
import torch
import torch.nn as nn
from data import generate_loader
from option import get_option

class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.dev = torch.device("cuda: {}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('', 'ghostnet', opt.num_classes, source='local').to(self.dev)

        if opt.multigpu:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.device_ids).to(self.dev)

        print("# params:", sum(map(lambda x: x.numel(), self.model.parameters())))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.dev)
        self.optim = torch.optim.RMSprop(self.model.parameters(), opt.lr, weight_decay=opt.weight_decay,
                                        alpha=0.9, eps=0.001, momentum=0.9)
        
        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.val_loader = generate_loader('val', opt)
        print("validation set ready")
        self.best_acc, self.best_epoch = 0, 0

    def fit(self):
        opt = self.opt
        print("start training")
        acc = []

        for epoch in range(opt.n_epoch):
            self.model.train()
            tqdm_message = "epoch " + str(epoch) + "/" + str(opt.n_epoch)
            for _, inputs in tqdm(enumerate(self.train_loader), desc=tqdm_message):
                images = inputs[0].to(self.dev)
                labels = inputs[1].to(self.dev)
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (epoch + 1) % opt.eval_epoch == 0:
                val_acc = self.eval(self.val_loader)
                acc.append(val_acc)

                if val_acc >= self.best_acc:
                    self.best_acc, self.best_epoch = val_acc, epoch
                    self.save(epoch+1)
                
                print("Epoch [{}/{}] Loss: {:.3f}, Test AccL {:.3f}".format(epoch+1, opt.n_epoch, loss.item(), val_acc))
                print("Best: {:.2f} @ {}".format(self.best_acc, self.best_epoch+1))

    @torch.no_grad()
    def eval(self, data_loader):
        loader = data_loader
        self.model.eval()
        num_correct, num_total = 0, 0

        for inputs in loader:
            images = inputs[0].to(self.dev)
            labels = inputs[1].to(self.dev)
            outputs = self.net(images)
            _, preds = torch.max(outputs.detach(), 1)

            num_correct += (preds == labels).sum().items()
            num_total += labels.size(0)
        
        return num_correct / num_total

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
        torch.save(self.net.state_dict(), save_path)



def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()


if __name__ == "__main__":
    main()