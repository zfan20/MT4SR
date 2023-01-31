# MT4SR
This is the implementation for the paper:
BigData'22. You may find it on [Arxiv](https://arxiv.org/pdf/2210.13572.pdf)

The code is built on Pytorch.
The dataset preprocess code is in data/ dir. Change the path in the code and also include the meta data file path.

Code to run:
```python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=relationsasrec_v6/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=RelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weigh=0.1 --outseq_rel_loss_weight=0.05```

Please cite our paper if you use the code:
```bibtex
@inproceedings{fan2022sequentialmt4sr,
  title={Sequential Recommendation with Auxiliary Item Relationships via Multi-Relational Transformer},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Chen and Huang, Peijie and Peng, Hao and Philip, S Yu},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={525--534},
  year={2022},
  organization={IEEE}
}
```
