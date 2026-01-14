```
#### Stage 1

python selor_amr/stage1/extract_triples.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --test_csv data/yelp_review_polarity_csv/test_with_amr.csv \
 --triples_col 2 \
 --csv_has_header False \
 --out_dir result/triples

#### Stage 2

python selor_amr/stage2/build_triple_pool.py \
 --triples_dir result/triples \
 --max_triples 50 \
 --min_freq 5 \
 --out_dir result/triples



python selor_amr/stage2/build_triple_pool.py \
    --triples_dir result/triples \
    --labels_csv data/yelp_review_polarity_csv/train_with_amr.csv \
    --max_triples 50 \
    --min_freq 3 \
    --filter_method chi2 \
    --top_k 30000 \
    --out_dir result/triples







#### Stage 3

python selor_amr/stage3/extract_cls_embedding.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --text_col 1 \
 --batch_size 32 \
 --gpu 0 \
 --out_path result/embeddings/train_cls.pt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python selor_amr/stage3/pretrain_ce_triple.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --csv_has_header False \
 --label_col 0 \
 --triples_dir result/triples \
 --train_embedding result/embeddings/train_cls.pt \
 --antecedent_len 3 \
 --num_samples 50000 \
 --min_coverage 2 \
 --batch_size 64 \
 --epochs 20 \
 --learning_rate 1e-4 \
 --weight_decay 1e-5 \
 --gpu 0 \
 --save_dir result/ce_triple \
 --seed 42 \
 --label_offset 1 \
 --valid_split 0.1

#### Stage 4

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python selor_amr/stage4/amr_selor.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --test_csv data/yelp_review_polarity_csv/test_with_amr.csv \
 --csv_has_header False \
 --text_col 1 \
 --label_col 0 \
 --label_offset 1 \
 --triples_dir result/triples \
 --train_embedding result/embeddings/train_cls.pt \
 --ce_path result/ce_triple/ce_triple_best.pt \
 --ce_config result/ce_triple/ce_triple_config.pkl \
 --antecedent_len 3 \
 --max_triples 50 \
 --batch_size 16 \
 --epochs 2 \
 --learning_rate 1e-4 \
 --weight_decay 1e-5 \
 --gpu 7 \
 --save_dir result/amr_selor \
 --seed 42

#### Stage 5 (Eval)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python selor_amr/stage5/eval_amr_selor.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --test_csv data/yelp_review_polarity_csv/test_with_amr.csv \
 --csv_has_header False \
 --text_col 1 \
 --label_col 0 \
 --label_offset 1 \
 --triples_dir result/triples \
 --train_embedding result/embeddings/train_cls.pt \
 --ce_path result/ce_triple/ce_triple_best.pt \
 --model_path result/amr_selor/amr_selor_best.pt \
 --antecedent_len 3 \
 --max_triples 50 \
 --batch_size 16 \
 --gpu 0 \
 --save_dir result/amr_selor_eval \
 --seed 42

#### Stage 6 (Pipeline Runner)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python selor_amr/stage6/run_amr_selor.py \
 --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
 --test_csv data/yelp_review_polarity_csv/test_with_amr.csv \
 --csv_has_header False \
 --text_col 1 \
 --label_col 0 \
 --label_offset 1 \
 --triples_dir result/triples \
 --emb_path result/embeddings/train_cls.pt \
 --ce_dir result/ce_triple \
 --amr_selor_dir result/amr_selor \
 --eval_dir result/amr_selor_eval \
 --antecedent_len 3 \
 --max_triples 50 \
 --batch_size 16 \
 --epochs 16 \
 --lr 1e-4 \
 --weight_decay 1e-5 \
 --gamma 0.95 \
 --gpu 0 \
 --seed 42 \
 --run_stage4 --run_stage5

#### Stage 6 (Inference-only)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python selor_amr/stage6/inference_amr_selor.py \
 --input_csv data/yelp_review_polarity_csv/test_with_amr.csv \
 --csv_has_header False \
 --text_col 1 \
 --triples_dir result/triples \
 --train_embedding result/embeddings/train_cls.pt \
 --ce_path result/ce_triple/ce_triple_best.pt \
 --model_path result/amr_selor/amr_selor_best.pt \
 --antecedent_len 3 \
 --max_triples 50 \
 --batch_size 32 \
 --gpu 0 \
 --seed 42 \
 --start_index 0 \
 --output_csv result/amr_selor_infer/predictions.csv
```

