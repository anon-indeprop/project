import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
from hp import hp
import numpy as np
from model import BertClassifier
from transformers import AdamW
from dataloader import PropDataset, pad, tag2idx, idx2tag, num_task, masking
import time
import random

SEED = 652
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.manual_seed(hp.seed)

timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

def train(model, iterator, optimizer, criterion, binary_criterion):
    model.train()

    train_losses = []

    for k, batch in enumerate(iterator):
        if hp.dummy:
            if k > 1:
                break
        words, x, is_heads, att_mask, tags, y, seqlens = batch
        att_mask = torch.Tensor(att_mask)
#        print(y)
        optimizer.zero_grad()
        logits, _ = model(x, attention_mask=att_mask)

        loss = []
        if masking or num_task  == 2:
            for i in range(num_task):
                logits[i] = logits[i].view(-1, logits[i].shape[-1])
            y[0] = y[0].view(-1).to(hp.device)
            y[1] = y[1].float().to(hp.device)
            loss.append(criterion(logits[0], y[0]))
            loss.append(binary_criterion(logits[1], y[1]))
        else:
            for i in range(num_task):
                logits[i] = logits[i].view(-1, logits[i].shape[-1]) # (N*T, 2)
                y[i] = y[i].view(-1).to(hp.device)
                loss.append(criterion(logits[i], y[i]))

        if num_task == 1:
            joint_loss = loss[0]
        elif num_task == 2:
            joint_loss = hp.alpha*loss[0] + (1-hp.alpha)*loss[1]
        joint_loss.backward()
        optimizer.step()
        train_losses.append(joint_loss.item())

        if k%10==0: # monitoring
            print("step: {}/{}, loss[0]: {}, joint_loss: {}".format(
                            k, len(iterator), loss[0].item(), joint_loss.item()
                ))

    train_loss = np.average(train_losses, axis=0)

    return train_loss

def eval(model, iterator, f, criterion, binary_criterion):
    print(f)
    model.eval()

    valid_losses = []

    Words, Is_heads = [], []
    Tags = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    Y_hats = [[] for _ in range(num_task)]
    with torch.no_grad():
        for k , batch in enumerate(iterator):
            if hp.dummy:
                if k > 1:
                    break
            words, x, is_heads, att_mask, tags, y, seqlens = batch
 #           print(x,att_mask)
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask) # logits: (N, T, VOCAB), y: (N, T)

            loss = []
            if num_task == 2 or masking:
                for i in range(num_task):
                    logits[i] = logits[i].view(-1, logits[i].shape[-1]) # (N*T, 2)
                y[0] = y[0].view(-1).to(hp.device)
                y[1] = y[1].float().to(hp.device)
                loss.append(criterion(logits[0], y[0]))
                loss.append(binary_criterion(logits[1], y[1]))
            else:
                for i in range(num_task):
                    logits[i] = logits[i].view(-1, logits[i].shape[-1]) # (N*T, 2)
                    y[i] = y[i].view(-1).to(hp.device)
                    loss.append(criterion(logits[i], y[i]))

            if num_task == 1:
                joint_loss = loss[0]
            elif num_task == 2:
                joint_loss = hp.alpha*loss[0] + (1-hp.alpha)*loss[1]

            valid_losses.append(joint_loss.item())

            Words.extend(words)
            Is_heads.extend(is_heads)

            for i in range(num_task):
                Tags[i].extend(tags[i])
                Y[i].extend(y[i].cpu().numpy().tolist())
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())
    valid_loss = np.average(valid_losses, axis=0) 

    with open(f, 'w') as fout:
        y_hats, preds = [[] for _ in range(num_task)], [[] for _ in range(num_task)]
        if num_task == 1:
            for words, is_heads, tags[0], y_hats[0] in zip(Words, Is_heads, *Tags, *Y_hats):
                for i in range(num_task):
                    y_hats[i] = [hat for head, hat in zip(is_heads, y_hats[i]) if head == 1]
                    preds[i] = [idx2tag[i][hat] for hat in y_hats[i]]
                fout.write(words.split()[0])
                fout.write("\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{} {} {} \n".format(w,t1,p_1))
                fout.write("\n")

        elif num_task == 2:
            false_neg = 0
            false_pos = 0
            true_neg = 0
            true_pos = 0
            for words, is_heads, tags[0], tags[1], y_hats[0], y_hats[1] in zip(Words, Is_heads, *Tags, *Y_hats):
                y_hats[0] = [hat for head, hat in zip(is_heads, y_hats[0]) if head == 1]
                preds[0] = [idx2tag[0][hat] for hat in y_hats[0]]
                preds[1] = idx2tag[1][y_hats[1]]

                if tags[1].split()[1:-1][0] == 'Non-prop' and preds[1] == 'Non-prop':
                    true_neg += 1
                elif tags[1].split()[1:-1][0] == 'Non-prop' and preds[1] == 'Prop':
                    false_pos += 1
                elif tags[1].split()[1:-1][0] == 'Prop' and preds[1] == 'Prop':
                    true_pos += 1
                elif tags[1].split()[1:-1][0] == 'Prop' and preds[1] == 'Non-prop':
                    false_neg += 1
                
                fout.write(words.split()[0])
                fout.write("\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{} {} {} {} {}\n".format(w,t1,tags[1].split()[1:-1][0],p_1,preds[1]))
                fout.write("\n")
            try:
                precision = true_pos / (true_pos + false_pos)
            except ZeroDivisionError:
                precision = 1.0
            try:
                recall = true_pos / (true_pos + false_neg)
            except ZeroDivisionError:
                recall = 1.0
            try:
                f1 = 2 *(precision*recall) / (precision + recall)
            except ZeroDivisionError:
                if precision*recall==0:
                    f1=1.0
                else:
                    f1=0.0
            print("sen_pre", precision)
            print("sen_rec", recall)
            print("sen_f1", f1)
            false_neg = false_pos = true_neg = true_pos = precision = recall = f1 = 0

    ## calc metric 
    y_true, y_pred = [], []
    for i in range(num_task):
        y_true.append(np.array([tag2idx[i][line.split()[i+1]] for line in open(f, 'r').read().splitlines() if len(line.split()) > 1]))
        y_pred.append(np.array([tag2idx[i][line.split()[i+1+num_task]] for line in open(f, 'r').read().splitlines() if len(line.split()) > 1]))
    
    num_predicted, num_correct, num_gold = 0, 0, 0
    if num_task != 2:
        for i in range(num_task):
            num_predicted += len(y_pred[i][y_pred[i]>1])
            num_correct += (np.logical_and(y_true[i]==y_pred[i], y_true[i]>1)).astype(np.int).sum()
            num_gold += len(y_true[i][y_true[i]>1])
    elif num_task == 2:  
        num_predicted += len(y_pred[0][y_pred[0]>1])
        num_correct += (np.logical_and(y_true[0]==y_pred[0], y_true[0]>1)).astype(np.int).sum()
        num_gold += len(y_true[0][y_true[0]>1])
        
    print("num_predicted:{}".format(num_predicted))
    print("num_correct:{}".format(num_correct))
    print("num_gold:{}".format(num_gold))
    try:
        precision = num_correct / num_predicted
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    final = f + ".P%.4f_R%.4f_F1%.4f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write("{}\n".format(result))

        fout.write("precision={:4f}\n".format(precision))
        fout.write("recall={:4f}\n".format(recall))
        fout.write("f1={:4f}\n".format(f1))

    os.remove(f)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1, valid_loss

if __name__=="__main__":
    if hp.wandb:
        import wandb
        wandb.init(project="classifier", name=hp.run, config=vars(hp))

    model = BertClassifier(hp)
    if hp.device == "cuda":
        print("Detect ", torch.cuda.device_count(), "GPUs!")
    else:
        print("\nUSING CPUS!")
    model = nn.DataParallel(model)
    model.to(hp.device)
    if hp.wandb:
        wandb.watch(model)

    train_dataset = PropDataset(hp.trainset, './dataset/new/train.json', IsTest=False)
    if hp.dummy:
        eval_dataset = train_dataset
    else:
        eval_dataset = PropDataset(hp.validset, './dataset/new/dev.json', IsTest=True)
    test_dataset = PropDataset(hp.testset, './dataset/new/test.json', IsTest=True)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=pad)

    warmup_proportion = 0.1
    num_train_optimization_steps = int(len(train_dataset) / hp.batch_size ) * hp.n_epochs
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #optimizer_grouped_parameters = [
    #    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #]

    optimizer = AdamW(model.parameters(),
                         lr=hp.lr)

    weights_labels_tensor = torch.tensor([0, 0.45] + [1] * 18).to(hp.device) # Check the order in data_load.py  of VOCAB ~line 40
    criterion = nn.CrossEntropyLoss(weight=weights_labels_tensor, ignore_index=0)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2000/14263]).to(hp.device))
    
    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=hp.patience, verbose=True)
    if hp.dummy:
        import pickle
        pickle.dump({'model':model, 'optimizer': optimizer, 'criterion': criterion, 'train_iter': train_iter,
                     'num_task': num_task, 'masking': masking, 'binary_criterion': binary_criterion},
                    open("../dump.pkl", 'wb'))

        for _ in range(0, 1000):
            train_loss = train(model, train_iter, optimizer, criterion, binary_criterion)
        exit()
    for epoch in range(1, hp.n_epochs+1):
        print("=========eval at epoch =",epoch,"=========")
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        if not os.path.exists('results'): os.makedirs('results')
        fname = os.path.join('checkpoints', timestr)
        spath = os.path.join('checkpoints', timestr+".pt")

        train_loss = train(model, train_iter, optimizer, criterion, binary_criterion)

        if hp.dummy:
            valid_loss = 0.0
        else:
            precision, recall, f1, valid_loss = eval(model, eval_iter, fname, criterion, binary_criterion)

        epoch_len = len(str(hp.n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.n_epochs:>{epoch_len}}]     ' +
                     'train_loss: ' + str(train_loss) +
                     ' valid_loss: ' + str(valid_loss))
        print(print_msg)

        if hp.wandb:
            wandb.log({"Training Loss": train_loss[0],
                        "Validation Loss": valid_loss[0],
                        "Training Joint Loss": train_loss[1],
                        "Validation Joint Loss": valid_loss[1],
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1})
        # early_stopping(-1*f1, model, spath)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    res = os.path.join('results', timestr)
    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load(spath))
    precision, recall, f1, test_loss = eval(model, test_iter, res, criterion, binary_criterion)
    print_msg = (f'test_precision: {precision:.5f} ' +
                 f'test_recall: {recall:.5f} ' +
                 f'test_f1: {f1:.5f}')
    print(print_msg)
