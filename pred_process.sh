#python predict.py -test data/sogou_valid.json \
python predict_my.py -test data/sogou_shuffle_valid.json \
-valid-data data/cross_valid-1.pt \
-device 6 \
-model saved_checkpoint/cross_1/cross_1.best.query.pre.model \
-output output/cross_1 \
-question \
