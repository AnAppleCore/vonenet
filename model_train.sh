#python train.py train --in_path /root/data/barrycao/data/imagenet -o results/squeezenet1_1/ --model_arch squeezenet1_1 --ngpus 4 --workers 4 --epochs 20 --restore_epoch 2 --restore_path results/squeezenet1_1/

python train.py train --in_path /root/data/barrycao/data/imagenet -o results/vgg19_bn/ --model_arch vgg19_bn --ngpus 4 --workers 4 --epochs 20 --batch_size 384 --restore_epoch 9 --restore_path results/vgg19_bn/
