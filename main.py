# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-09-05 23:05:26

from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from utils.data import Data
try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


save_file = "result_7.10.csv"


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0


    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, input_label_seq_tensor = batchify_with_label(instance, data.HP_gpu, data.label_alphabet_size)
        #batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask ,input_label_seq_tensor = batchify_with_label(instance, data.HP_gpu,data.label_alphabet_size ,True)
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest,input_label_seq_tensor)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            #batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask,input_label_seq_tensor
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask,input_label_seq_tensor)


        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)

        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu,label_size = 20, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    #label_instance
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    #
    input_label_seq_tensor = autograd.Variable(torch.zeros((batch_size, label_size)),volatile =  volatile_flag).long()

    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long())
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        input_label_seq_tensor[idx, :label_size] = torch.LongTensor([i for i in range(label_size)])
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])


    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        input_label_seq_tensor = input_label_seq_tensor.cuda()
        mask = mask.cuda()


    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask ,input_label_seq_tensor


def train(data):
    print("Training model...")
    writer = SummaryWriter('runs')
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    model = SeqModel(data)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print ("--------pytorch total params--------")
    print (pytorch_total_params)

    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=data.HP_lr,
                              momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    best_test = -10
    best_epoch = -1
    no_imprv_epoch = 0
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))        #print (self.train_Ids)
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)

        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]

            if not instance:
                continue

            #label_instance = [[i for i in range(0, data.label_alphabet_size + 1)] for _ in range(len(instance))]

            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask ,input_label_seq_tensor= batchify_with_label(instance, data.HP_gpu,data.label_alphabet_size)

            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask,input_label_seq_tensor)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            if end%500 == 0:
                # temp_time = time.time()
                # temp_cost = temp_time - temp_start
                # temp_start = temp_time
                #print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")


                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            if data.whether_clip_grad:
                from torch.nn.utils import clip_grad_norm
                clip_grad_norm(model.parameters(), data.clip_grad)
            optimizer.step()
            model.zero_grad()
        writer.add_scalar('Train', total_loss, idx)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            #exit(1)
        # continue
        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        writer.add_scalar('Dev', acc, idx)

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))


        # ## decode test
        speed, acc_test, p, r, f_test, _,_ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc_test, p, r, f_test))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc_test))

        writer.add_scalar('Test', acc_test, idx)

        if current_score > best_dev:
            if data.seg:
                best_test = f_test
                print("Exceed previous best f score:", best_dev)
            else:
                best_test = acc_test
                print("Exceed previous best acc score:", best_dev)
            best_epoch = idx
            # model_name = data.model_dir +'.'+ str(idx) + ".model"
            # print("Save current best model in file:", model_name)
            # torch.save(model.state_dict(), model_name)
            best_dev = current_score
            no_imprv_epoch = 0


        else:
            #early stop
            no_imprv_epoch += 1
            if no_imprv_epoch >= 10:
                print("early stop")
                print("Current best f score in dev", best_dev)
                print("Current best f score in test", best_test)
                break

        if data.seg:
            print ("Current best f score in dev",best_dev)
            print ("Current best f score in test",best_test)
        else:
            print ("Current best acc score in dev",best_dev)
            print ("Current best acc score in test",best_test)
        gc.collect()

    writer.close()
    import csv
    # with open("without_embedd.csv", "a", encoding='utf8', newline='') as c:
    #     writer = csv.writer(c)
    #     writer.writerow([data.label_embedding_scale,data.clip_grad,str(data.use_crf),data.HP_momentum,data.HP_hidden_dim,data.HP_lstm_layer,data.HP_lr,data.HP_lr_decay,data.HP_dropout,best_epoch,best_dev,best_test])
    with open(save_file, "a", encoding='utf8', newline='') as c:
        writer = csv.writer(c)
        writer.writerow([best_epoch,best_dev,best_test])



def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File' )
    # # POS wsj
    parser.add_argument('--train_dir', default='wsj_pos/train.pos', help='train_file')
    parser.add_argument('--dev_dir', default='wsj_pos/dev.pos', help='dev_file')
    parser.add_argument('--test_dir', default='wsj_pos/test.pos', help='test_file')
    parser.add_argument('--model_dir', default='wsj_pos/label_embedding', help='model_file')
    parser.add_argument('--seg', default=False)


    #embeddings-scaled.EMBEDDING_SIZE=50.txt
    #parser.add_argument('--word_emb_dir', default='embeddings-scaled.EMBEDDING_SIZE=50.txt', help='word_emb_dir')
    parser.add_argument('--word_emb_dir', default='glove.6B.100d.txt', help='word_emb_dir')
    #parser.add_argument('--word_emb_dir', default='', help='word_emb_dir')
    parser.add_argument('--norm_word_emb', default = False)
    parser.add_argument('--norm_char_emb', default = False)
    parser.add_argument('--number_normalized', default = True)
    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--char_emb_dim', default=30)

    #NetworkConfiguration
    parser.add_argument('--use_crf', default= False)
    parser.add_argument('--use_char', default=True)
    parser.add_argument('--word_seq_feature', default='LSTM')
    parser.add_argument('--char_seq_feature', default='LSTM')



    #TrainingSetting
    parser.add_argument('--status', default='train')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--iteration',default = 100)
    parser.add_argument('--batch_size', default= 10)
    parser.add_argument('--ave_batch_loss', default=False)

    #Hyperparameters
    parser.add_argument('--cnn_layer', default=4)
    parser.add_argument('--char_hidden_dim', default=50)
    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--lstm_layer', default=0)
    parser.add_argument('--bilstm', default=True)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--lr_decay',default=0.05)
    parser.add_argument('--label_embedding_scale',default = 0.0025)
    parser.add_argument('--num_attention_head', default=5)
    #0.05
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--whether_clip_grad', default=True)
    parser.add_argument('--clip_grad', default=5)
    parser.add_argument('--l2', default=1e-8)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--seed',default=42)


    args = parser.parse_args()

    print (args.seg)

    seed_num = int(args.seed)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    data = Data()
    #print(data.initial_feature_alphabets())
    data.HP_gpu = torch.cuda.is_available()
    data.read_config(args)
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")