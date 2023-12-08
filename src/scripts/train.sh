device="cuda:2"

# for model in resnet18 vgg16 mobilenetv2 wrn16_4 googlenet inception_v3 vit
# do
#     python -u train.py --model ${model} --device ${device} > ../weights/${model}.log
# done
for model in resnet50
do
    python -u train.py --model ${model} --device ${device} > ../weights/${model}.log
done