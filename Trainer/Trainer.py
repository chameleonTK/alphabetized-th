import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, device, model, datasets, evaluator, lr=0.002):
        self.model = model.to(device)
        self.datasets = datasets
        self.evaluator = evaluator    
        
        self.optimizer = None
        self.criterion = None


    def forward(self, batch, model, criterion):
        pass

    def train(self, epochs, **kwargs):
        eval_sample = None if "eval_sample" not in kwargs else kwargs["eval_sample"]
        log = None if "log" not in kwargs else kwargs["log"]
        save_path = "./Files/best_model.pt" if "save_path" not in kwargs else kwargs["save_path"]

        train_iterator = self.datasets.train_iterator
        valid_iterator = self.datasets.valid_iterator
        test_iterator = self.datasets.test_iterator

        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        evaluator = self.evaluator

        val_loss, acc = evaluator.run(model, valid_iterator, criterion, eval_sample=eval_sample)
        print("Validation", "START", acc, val_loss)
        best_acc = acc
        torch.save(model.state_dict(), save_path)

        iterations = 0
        start = time.time()
        
        train_stat = []
        dev_every = max(10, int(epochs*len(train_iterator)//100))
        dev_every = min(dev_every, 1000)

        print("Epochs", epochs)
        print("#Batch", len(train_iterator))
        print("Batch size", self.datasets.batch_size)

        with tqdm(total=epochs*len(train_iterator)) as pbar:
        
            for epoch in range(epochs):
                train_iterator.init_epoch()

                n_correct, n_total, train_loss = 0, 0, 0
                for batch_idx, batch in enumerate(train_iterator):
                    # switch model to training mode, clear gradient accumulators
                    model.train()
                    optimizer.zero_grad()

                    iterations += 1

                    # forward pass
                    loss = self.forward(batch, model, criterion)
                    loss.backward()

                    optimizer.step()

                    train_loss += loss.item()

                    stat = {
                        "epoch": epoch,
                        "step": iterations,
                        "train_loss": loss.item()
                    }

                    if iterations > 0 and iterations % dev_every == 0:
                        val_loss, acc = evaluator.run(model, valid_iterator, criterion)
                        print("Validation", iterations, acc, val_loss)
                        if acc > best_acc:
                            best_acc = acc
                            torch.save(model.state_dict(), save_path)

                        last_val_iter = iterations
                        stat["val_loss"] = val_loss
                        stat["eval"] = acc.get_values()
                        stat["best_eval"] = best_acc.get_eval_value()
                        stat["time"] = (time.time() - start)

                        
                    if log is not None:
                        log.log({'train_loss': stat["train_loss"]}, stat)
                            

                    train_stat.append(stat)
                    pbar.update(1)
        
        model.load_state_dict(torch.load(save_path))
        return model, train_stat
