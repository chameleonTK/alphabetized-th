import numpy as np
import random
import torch
import os
import importlib
from pathlib import Path
from Datasets.ThDataset import ThDataset

class Experiment:
    def __init__(self, outputDir, batch_size=16):
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        
        self.min_freq = 1
        self.batch_size = batch_size

        self.init_token = None
        self.eos_token = None
        self.model_type = None
    
    def set_random_seed(self, seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)    
        np.random.seed(seed)
        np.random.RandomState(seed)

        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) #seed all gpus    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False  
        torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def load_object_by_class(config):
        _module = importlib.import_module(config["module"])
        _class = getattr(_module, config["class"])
        _instance = _class(**config["kwargs"])

        return _instance
    
    @classmethod
    def load(expcls, dir):
        dataset = ThDataset("", name="")
        dataset.load_fields(f"{dir}/datasets.jsonl")

        configs = ThDataset.load_jsonl(f"{dir}/model_config.jsonl")
        model_configs = configs[0]
        model = Experiment.load_object_by_class(model_configs)
        model.load_state_dict(torch.load(f"{dir}/model.pt"))
        model.eval()


        eval_configs = configs[1]
        eval_configs["kwargs"]["datasets"] = dataset
        evaluator = Experiment.load_object_by_class(eval_configs)

        if len(configs) >= 3:
            dataset_configs = configs[2]
            dataset_configs["tokenizer"]["kwargs"]["dataset"] = dataset
            tokenizer = Experiment.load_object_by_class(dataset_configs["tokenizer"])
            dataset_configs["kwargs"]["dataset"] = dataset
            dataset_configs["kwargs"]["tokenizer"] = tokenizer
            
            dataset = Experiment.load_object_by_class(dataset_configs)
            dataset.load_fields(f"{dir}/datasets.jsonl")
        

        exp = expcls(dir, dataset)
        exp.stats = {
            "model": model,
            "evaluator": evaluator,
            "trainer": None,
            "datasets": dataset,
        }
        
        return exp
    
    def load_datasets(self, MIN_FREQ, BATCH_SIZE, **kwargs):
        weighted_loss = False if "weighted_loss" not in kwargs else kwargs["weighted_loss"]

        nclass = None
        datasets = self.datasets
        acc_cnt, cnt = datasets.describe()
        if weighted_loss:
            nclass = {k:acc_cnt-v for (k,v) in cnt.items()}

        datasets.build(self.device, MIN_FREQ = MIN_FREQ, BATCH_SIZE = BATCH_SIZE, init_token=self.init_token, eos_token=self.eos_token)
        datasets.save_fields(f"{self.outputDir}/datasets.jsonl")

        print(f"No. TOKEN: {len(datasets.fields['tokens'].vocab)}")
        print(f"No. LABEL: {len(datasets.fields['labels'].vocab)}")
        print(datasets.fields['labels'].vocab.itos)
        # print(datasets.fields['tokens'].vocab.itos)
        datasets.save_jsonl(f"{self.outputDir}/labels.jsonl", datasets.fields['labels'].vocab.stoi)

        return datasets, nclass
    
    def run(self, epoch, **kwargs):
        print("== START ==")
        self.set_random_seed()
        
        log = None if "log" not in kwargs else kwargs["log"]
        embed_weights = None if "embed_weights" not in kwargs else kwargs["embed_weights"]

        self.kwargs = kwargs

        print("== LOAD DATASET ==")
        datasets, nclass = self.load_datasets(self.min_freq, self.batch_size, **kwargs)
        dataset_config = datasets.get_config()

        print("== INIT MODEL ==")
        model, model_config = self.init_model(embed_weights, datasets, **kwargs)

        print("== TRAIN ==")
        evaluator, eval_config = self.init_evaluator(datasets, **kwargs)
        trainer = self.init_trainer(self.device, model, datasets, evaluator, nclass=nclass, **kwargs)
        
        print("Start training...")
        model, train_stat = trainer.train(epoch, **kwargs)

        print("== EVAL ==")
        print("Testing")
        test_loss, acc, results = evaluator.run(model, datasets.test_iterator, trainer.criterion, return_pred=True)

        print("Saving model...")
        datasets.save_jsonl(f"{self.outputDir}/model_config.jsonl", [model_config, eval_config, dataset_config])
        print(f"{self.outputDir}/model.pt")
        torch.save(model.state_dict(), f"{self.outputDir}/model.pt")
        
        print("Saving stat...")
        train_stat.append({"epoch": epoch, "step": -1, "test_result": acc.get_values()})
        datasets.save_jsonl(f"{self.outputDir}/train_stat.jsonl", train_stat)
        
        print("Prepraring results...")
        TOKEN_PAD_IDX = datasets.fields["tokens"].vocab.stoi[datasets.fields["tokens"].pad_token]
        outputs = self.get_outputs(datasets, results, TOKEN_PAD_IDX)
        
        print("Saving results...")
        datasets.save_jsonl(f"{self.outputDir}/results.jsonl", outputs)

        print("Test:", acc)

        self.stats = {
            "model": model,
            "evaluator": evaluator,
            "trainer": trainer,
            "datasets": datasets,
        }

        return model, results, train_stat, (evaluator, trainer)