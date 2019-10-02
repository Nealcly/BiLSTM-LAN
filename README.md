# BiLSTM - Label Attention Network (BiLSTM-LAN)
 [Hierarchically-Refined Label Attention Network for Sequence Labeling](https://arxiv.org/pdf/1908.08676.pdf) (EMNLP 2019)
 

# Model Structure

The model consists of 2 BiLSTM-LAN layers. Each BiLSTM-LAN layer is composed of a BiLSTM encoding sublayer and a label-attention inference sublayer.  In paticular, the former is the same as the BiLSTM layer in the baseline model, while the latter uses multihead attention (Vaswani et al., 2017) to jointly encode information from the word representation subspace and the label representation subspace.

<img src="model.jpg" width="1000" >


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

# Cite
```
Leyang Cui and Yue Zhang Hierarchically-Refined Label Attention Network for Sequence Labeling. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP) and 9th International Joint Conference on Natural Language Processing (IJCNLP) in Hong Kong, China.

```

# Acknowledgments
[NCRF++](https://github.com/jiesutd/NCRFpp)
