device="cuda:0"


load_path=../weights
mkdir -p ../results/images/images
python -u attack.py --load_path ${load_path} --device ${device} >> ../test/mix.log