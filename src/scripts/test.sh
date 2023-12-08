device="cuda:0"

for model in vit-
do
    load_path=xxx/vision-transformers-cifar10/checkpoint/vit-.pth
    python -u test.py --model ${model} --load_path ${load_path} --device ${device} >> ../test/${model}.log
done