V1
preprocess.py
python bin/preprocess.py --input data/yelp_review_polarity_csv/train.csv --output data/yelp_review_polarity_csv/train_with_index.csv
===
python bin/preprocess.py \
    --input data/yelp_review_polarity_csv/train.csv \
    --output data/yelp_review_polarity_csv/train_with_index.csv


prepare_yelp_to_txt
python bin/prepare_yelp_to_txt.py --csv data/yelp_review_plarity_csv/train_with_index.csv --out data/yelp_review_polarity_csv/train.sent.txt --split-sentences --max-sent-per-review 2,16 --review-id-col 0
===
python bin/prepare_yelp_to_txt.py \
    --csv data/yelp_review_polarity_csv/train_with_index.csv \
    --out data/yelp_review_polarity_csv/train.sent.txt \
    --split-sentences \
    --max-sent-per-review 3,10 \
    --review-id-col 0


predict_amrs_from_plaintext
python bin/predict_amrs_from_plaintext.py \
    --texts data/yelp_review_polarity_csv/train.sent.txt \
    --maps data/yelp_review_polarity_csv/train.sent.txt.map.tsv \
    --checkpoint checkpoints/AMR2.parsing.pt \
    --beam-size 5 \
    --batch-size 64 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --shard-open "[12, 2, 2, 2, 3, 3, 3, 4, 4, 4, 7, 7, 7]" \
    --continue False

python bin/predict_amrs_from_plaintext.py \
    --texts data/yelp_review_polarity_csv/train.sent.txt \
    --maps data/yelp_review_polarity_csv/train.sent.txt.map.tsv \
    --checkpoint checkpoints/AMR2.parsing.pt \
    --beam-size 1 \
    --batch-size 1 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --shard-open "[12, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]" \
    --continue True


finish_end_Sentence.py
python bin/finish_end_Sentence.py \
    --text data/yelp_review_polarity_csv/train.sent.txt \
    --shard-open "[12, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]"


check.py
python bin/check.py --csv result/train.sent.txt.csv
python bin/check.py --csv outputs/train_amr_triples.csv


amr_to_instance.py
python sjx/amr_to_instance.py \
    --input-csv result/train.sent.txt.csv \
    --output outputs/train_amr_triples.csv \
    --roles [0,1,2,3] \
    --extra-roles "mod,manner,cause,time,location"


merge_triplese.py
python sjx/merge_triplese.py \
    --amr-csv outputs/train_amr_triples.csv \
    --output outputs/train_amr_triples_essay.csv


triple_to_essay_id.py
python sjx/triple_to_essay_id.py \
    --train-csv data/yelp_review_polarity_csv/train_with_index.csv \
    --triples-csv outputs/train_amr_triples_essay.csv \
    --output outputs/train_with_triples.csv \
    --id-column 0