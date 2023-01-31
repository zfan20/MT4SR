# MT4SR
This is the implementation for the paper:
BigData'22. You may find it on [Arxiv](https://arxiv.org/pdf/2210.13572.pdf)

The code is built on Pytorch.
The dataset preprocess code is in data/ dir. Change the path in the code and also include the meta data file path.

Code to run:
```python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=relationsasrec_v6/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=RelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weigh=0.1 --outseq_rel_loss_weight=0.05```
