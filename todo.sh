cd ../ridge_segmentation/
python -u train.py --split_name 1
python -u segment_test.py --split_name 1
cd ../SentenceROP/
python -u cleansing.py --split_name 1
python -u train.py --split_name 1

cd ../ridge_segmentation/
python -u train.py --split_name 2
python -u segment_test.py --split_name 2
cd ../SentenceROP/
python -u cleansing.py --split_name 2
python -u train.py --split_name 2

cd ../ridge_segmentation/
python -u train.py --split_name 3
python -u segment_test.py --split_name 3
cd ../SentenceROP/
python -u cleansing.py --split_name 3
python -u train.py --split_name 3

cd ../ridge_segmentation/
python -u train.py --split_name 4
python -u segment_test.py --split_name 4
cd ../SentenceROP/
python -u cleansing.py --split_name 4
python -u train.py --split_name 4

python ring.py
shutdown