from Data.dataUtils import split_quora, split_mscoco, get_vocab, ger_stage_one_mscoco

# tokenize data, get vocab and split train dev
split_quora('./train.csv', out_path='data/quora/')
split_mscoco('captions_train2017.json', out_path='data/quora/')
# get vocab
get_vocab('data/mscoco/', dataset='mscoco')
get_vocab('data/quora/', dataset='quora')

# stage one data for mscoco
# quora already has negtive examples for stage one in original dataset
ger_stage_one_mscoco(inpath='data/mscoco/', out_path='data/mscoco/is_paraphrase/')



