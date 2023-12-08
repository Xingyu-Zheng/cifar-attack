# adversarial training!

device="cuda:1"

# for model in resnet18 vgg16 mobilenetv2 wrn16_4 googlenet inception_v3
# do
#     python -u train.py --model ${model} --device ${device} > ../weights/${model}.log
# done
for model in resnet18
do
    python -u train_pgd.py --model ${model} --device ${device} > ../weights/pgd/${model}.log
done