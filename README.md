# BiLSTM - Label Attention Network
 Hierarchically-Refined Label Attention Network for Sequence Labeling (EMNLP 2019)
 
# Requirement
* Python3
* PyTorch: 0.3

# Train models
* Download data and word embedding
* Run the script:
```
python main.py --learning_rate 0.01 --lr_decay 0.035 --dropout 0.5 --hidden_dim 400 --lstm_layer 3 --momentum 0.9 --whether_clip_grad True --clip_grad 5.0 \
--train_dir 'wsj_pos/train.pos' --dev_dir 'wsj_pos/dev.pos' --test_dir 'wsj_pos/test.pos' --model_dir 'wsj_pos/' --word_emb_dir 'glove.6B.100d.txt'
```

## Performance


|ID| TASK | Dataset |Performace
|---|--------- | --- | ---
|1| POS | wsj | 97.65 |
|2| POS | UD v2.2 | 95.59
|3| NER |  OntoNotes 5.0 | 88.16
|4| CCG |  CCGBank | 94.7

# Acknowledgments
[NCRF++](https://github.com/jiesutd/NCRFpp)