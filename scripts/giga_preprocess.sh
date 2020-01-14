python preprocess.py -train_src ~/links/data/giga/train.halfsplit.art.pt2.txt \
    -train_tgt ~/links/data/giga/train.halfsplit.tgt.pt2.txt \
    -valid_src ~/links/data/giga/valid.earlystop.art.txt \
    -valid_tgt ~/links/data/giga/valid.earlystop.tgt.txt \
    -save_data ~/links/data/giga/giga_pt2/data/giga_halfsplit_pt2 \
    -src_seq_length 10000 \
    -dynamic_dict \
    -share_vocab \
    -shard_size 100000
