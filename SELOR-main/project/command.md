python selor_amr/stage1/extract_triples.py \
  --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
  --test_csv  data/yelp_review_polarity_csv/test_with_amr.csv \
  --triples_col 2 \
  --csv_has_header False \
  --out_dir result/triples


python selor_amr/stage2/build_triple_pool.py \
  --triples_dir result/triples \
  --max_triples 50 \
  --min_freq 5 \
  --out_dir result/triples 

