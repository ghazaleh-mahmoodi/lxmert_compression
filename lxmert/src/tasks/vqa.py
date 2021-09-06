# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import json
from posix import WSTOPSIG

import numpy as np
from numpy.lib.function_base import select
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
import lottery_ticket_hypothesis as LTH

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=int(args.num_workers),
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        print(args.train, args.batch_size)
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, file_name, args):
        
        train_result = {}
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        if not args.pruning:
            self.save('initial_weights')

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            
            train_result["Epoch %d Train" % (epoch)] = evaluator.evaluate(quesid2ans) * 100.
            
            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score

                    if args.pruning:
                        self.save(f"{file_name}_retrain_BEST")
                    else:
                        self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                train_result["Epoch %d Valid" % (epoch)] = valid_score * 100.
                train_result["Epoch %d Best" % (epoch)] = best_valid * 100.

            log_str += "Training for "+ self.output

            print(log_str, end='')
            with open(f"logs/{self.output.replace('/', '_')}.log", 'a') as f:
                f.write(log_str)
                f.flush()

        # if args.pruning:
        #      self.save(f"{file_name}_retrain_LAST")
        # else:
        #     self.save("LAST")

        return train_result

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def prune(self, finetune_weights_path, file_name, sparcity, args, PRUNING_MODE, initial_finetune_weights_path='initial_weights'):
        
        dict_result = {}

        log_str = "============ Start pruning ============\n"
        log_str += f"Pruning {finetune_weights_path}.pth\n"
        
        print("finetune_weights_path", finetune_weights_path)
        self.load(finetune_weights_path)
        model_score = self.evaluate(self.valid_tuple)* 100.
        
        print(f"Initial score: {model_score}")
        log_str += f"Initial score: {model_score}\n"
        dict_result['Initial score'] = model_score
        
        
        #Low Magnitude Pruning
        new_weights, mask, logs_result, _ = LTH.low_magnitude_pruning(self, sparcity, args, dict_result, file_name)
        if(PRUNING_MODE == 'Low_Magnitude'):
            log_str += logs_result
            log_str += "convert mask to low magnitude pruning.\n"
            params_file = f"{self.output}/{file_name}_mask.npz"
            np.savez_compressed(params_file, mask)
            log_str += f"save mask in {self.output}/{file_name}_mask.npz.\n" 
        
        elif(PRUNING_MODE == 'High_Magnitude'):
            mask = LTH.high_magnitude_pruning(mask)
            log_str += "convert mask to high magnitude pruning.\n" 
            params_file = f"{self.output}/{file_name}_mask.npz"
            np.savez_compressed(params_file, mask)
            log_str += f"save mask in {self.output}/{file_name}_mask.npz.\n" 
        
        elif(PRUNING_MODE == 'Random'):
            mask = LTH.get_random_subnet(mask)
            params_file = f"{self.output}/{file_name}_mask.npz"
            np.savez_compressed(params_file, mask)
            log_str += f"save mask in {self.output}/{file_name}_mask.npz.\n" 
            log_str += "convert mask to random pruning.\n"
        
        if initial_finetune_weights_path is not None:
            log_str += "Reset to initial weights based on the mask.\n"

            new_weights = torch.load(self.output.replace('pruning', 'finetuned')+"/initial_weights.pth")
            new_weights = LTH.apply_mask(new_weights, mask)
        
        self.model.load_state_dict(new_weights)
        
        log_str += f"accuarcy is {self.evaluate(self.valid_tuple)* 100.}\n"
        log_str += "============ End of pruning ============\n"
        
        dict_result['accuarcy after pruning'] = self.evaluate(self.valid_tuple) * 100.
        
        self.save(file_name)
        
        with open(os.path.join(self.output, "%s.log" % file_name), 'a') as f:
            f.write(log_str)
            f.flush()
        
        return dict_result

if __name__ == "__main__":
    experiment_result = {}

    # Build Class
    torch.cuda.empty_cache()
    vqa = VQA()
    
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)
    
    print("args.pruning", args.pruning)
    
    if args.pruning:

        FINETUNE_MODEL_PATH = vqa.output.replace('pruning', 'finetuned')+'/BEST'
        file_name = f"{args.pruningmode}-{args.sparsity}_percentage"
        print(file_name)
        experiment_result['pruning_result'] = vqa.prune(FINETUNE_MODEL_PATH, file_name, int(args.sparsity), args, args.pruningmode)
        
        print("experiment_result : ", experiment_result)
        
        vqa.load(args.output+"/"+file_name)
        experiment_result['retrain_result'] = vqa.train(vqa.train_tuple, vqa.valid_tuple, file_name, args)
        os.system(f'rm {args.output+"/"+file_name+".pth"}')
        with open(os.path.join(vqa.output, f"{file_name}.json"), "w", encoding="utf8") as json_file :
            json.dump(experiment_result, json_file, ensure_ascii=False, indent=4)

    else:
        # Test or Train
        if args.test is not None:
            args.fast = args.tiny = False       # Always loading all data in test
            test_file_result_name = ""
            if args.sparsity != '-1' :
                test_file_result_name = f"{args.pruningmode}-{args.sparsity}_percentage_"
            if 'test' in args.test:
                vqa.predict(
                    get_data_tuple(args.test, bs=950,
                                shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, f'{test_file_result_name}test_predict.json')
                )
            elif 'val' in args.test:    
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = vqa.evaluate(
                    get_data_tuple('minival', bs=950,
                                shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'minival_predict.json')
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test
        else:
            print('Splits in Train data:', vqa.train_tuple.dataset.splits)
            if vqa.valid_tuple is not None:
                print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
                print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
            else:
                print("DO NOT USE VALIDATION")
            
            train_result = vqa.train(vqa.train_tuple, vqa.valid_tuple, "", args)
            
            with open(os.path.join(vqa.output, "finetune_result.json"), "w", encoding="utf8") as json_file :
                json.dump(train_result, json_file, ensure_ascii=False, indent=4)