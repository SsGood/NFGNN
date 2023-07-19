import random
import argparse
import torch
import uuid
import pickle
import numpy as np
import torch.optim as optim
import scipy.sparse as sp
import torch.nn.functional as F
from networks import NFGNN
from utils import accuracy
import sys
import gc

class datasest_class:
    pass

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# def OGB_evaluator(out, labels, dataname):
#     ogb_evaluator = Evaluator(f'ogbn-{dataname}')
#     pred = out.argmax(1, keepdim=True)
#     input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
#     return ogb_evaluator.eval(input_dict)["acc"]

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def load_data(data_path, name, K):
    with open(data_path+"training_"+name+".pickle","rb") as fopen:
        train_data = pickle.load(fopen)

    with open(data_path+"validation_"+name+".pickle","rb") as fopen:
        valid_data = pickle.load(fopen)

    with open(data_path+"test_"+name+".pickle","rb") as fopen:
        test_data = pickle.load(fopen)

    with open(data_path+"labels_"+name+".pickle","rb") as fopen:
        labels = pickle.load(fopen)
        
    train_data = [mat for mat in train_data[:K+1] ]
    valid_data = [mat for mat in valid_data[:K+1] ]
    test_data  = [mat for mat in test_data[:K+1] ]
    train_labels = labels[0].reshape(-1).long()
    valid_labels = labels[1].reshape(-1).long()
    test_labels = labels[2].reshape(-1).long()
    
    dataset = datasest_class()
    dataset.num_features = train_data[0].shape[1]
    dataset.num_classes = max(int(train_labels.max()) + 1, int(valid_labels.max()) + 1, int(test_labels.max()) + 1)
    
    return dataset, [train_data, valid_data, test_data], [train_labels, valid_labels, test_labels]

def create_batch(input_data, batch_size):
    num_sample = input_data[0].shape[0]
    list_bat = []
    for i in range(0,num_sample,batch_size):
        if (i+batch_size)<num_sample:
            list_bat.append((i,i+batch_size))
        else:
            list_bat.append((i,num_sample))
    return list_bat

def train(model, data, label, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    acc_train = accuracy(output, label)
    loss_train = F.nll_loss(output, label)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()

def validate(model, data, label):
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_val = F.nll_loss(output, label)
        acc_val = accuracy(output, label, batch=True)
        return loss_val.item(),acc_val.item()

def test(model, data, label):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model(data)
        loss_test = F.nll_loss(output, label)
        acc_test = accuracy(output, label, batch=True)
        return loss_test.item(),acc_test.item()





def main(args_str=None):
# Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--data_path', type=str, default='../data/pre/')
    parser.add_argument('--dataname', type=str, default="products", help='datasets.')

    # parser.add_argument('--dev', type=int, default=0, help='device id')
    parser.add_argument('--net', type=str, default="ChebNetII", help='device id')
    parser.add_argument('--batch_size',type=int, default=102400, help='Batch size')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=300, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--ppnp', type=str, default='NF_prop')

    parser.add_argument('--pro_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--pro_wd', type=float, default=0.00005, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()
    print(args)
    
    set_rng_seed(args.seed)
    batch_size = args.batch_size
    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    dataset, data_x, data_y = load_data(args.data_path, args.dataname, args.K)
    [train_data, valid_data, test_data] = data_x
    [train_labels, valid_labels, test_labels] = data_y

    checkpt_file = f'./pretrained/{args.dataname}/'+uuid.uuid4().hex+'.pt'
    print(f'---------------------{checkpt_file}----------------------')

    model = NFGNN(dataset, args).to(device)
    optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin3.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': args.pro_wd, 'lr': args.pro_lr}])


    list_bat_train = create_batch(train_data, args.batch_size)
    list_bat_val = create_batch(valid_data, args.batch_size)
    list_bat_test = create_batch(test_data, args.batch_size)
    
    train_data = torch.stack(train_data, dim=0)
    valid_data = torch.stack(valid_data, dim=0)
    test_data  = torch.stack(test_data, dim=0)
    print(train_data.shape)

    count = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    valid_num = valid_data[0].shape[0]
    test_num = test_data[0].shape[0]
    for epoch in range(args.epochs):
        list_loss = []
        list_acc = []
        random.shuffle(list_bat_train)
        for st,end in list_bat_train:
            loss_tra,acc_tra = train(model, 
                                     train_data[:,st:end,:].to(device), 
                                     train_labels[st:end].to(device),
                                     optimizer)
            list_loss.append(loss_tra)
            list_acc.append(acc_tra)
        loss_tra = np.round(np.mean(list_loss),4)
        acc_tra = np.round(np.mean(list_acc),4)

        list_loss_val = []
        list_acc_val = []
        for st,end in list_bat_val:
            loss_val,acc_val = validate(model, 
                                        valid_data[:,st:end,:].to(device), 
                                        valid_labels[st:end].to(device))
            list_loss_val.append(loss_val)
            list_acc_val.append(acc_val)

        loss_val = np.round(np.mean(list_loss_val),4)
        acc_val = np.round((np.sum(list_acc_val))/valid_num,4)

        if epoch%10==0:
#             print('train_acc:',acc_tra,'>>>>>>>>>>train_loss:',loss_tra)
#             print('val_acc:',acc_val,'>>>>>>>>>>>val_loss:',loss_val)
            print(f'epoch: {epoch}/{args.epochs} trn_acc: {acc_tra} trn_loss: {loss_tra} val_acc: {acc_val} val_loss: {loss_val}')

        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            count = 0
        else:
            count += 1

        if count == args.early_stopping:
            break

    list_loss_test = []
    list_acc_test = []
    model.load_state_dict(torch.load(checkpt_file))
    
    for st,end in list_bat_test:
        loss_test,acc_test = test(model, 
                                  test_data[:,st:end,:].to(device), 
                                  test_labels[st:end].to(device))
        list_loss_test.append(loss_test)
        list_acc_test.append(acc_test)
    acc_test = (np.sum(list_acc_test))/test_num

    print(args.dataname)
    print('Load {}th epoch'.format(best_epoch))
    print(f"Valdiation accuracy: {np.round(acc*100,2)}, Test accuracy: {np.round(acc_test*100,2)}")
    
    return np.round(acc*100,2), np.round(acc_test*100,2), checkpt_file

if __name__ == '__main__':
    with RedirectStdStreams(stdout=sys.stderr):
        val_acc_mean, test_acc_mean, checkpt_file = main()
    print('%.4f,%.4f,%s' % (val_acc_mean, test_acc_mean, checkpt_file))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()









