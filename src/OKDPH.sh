python OKDPH.py --dataset cifar10 --model_names resnet32 resnet32 --runs 5 --log okdph_resnet32 \
    --omega 0.8 --beta 0.8 --gamma 0.5 --interval 1_epoch

python OKDPH.py --dataset cifar10 --model_names vgg16 vgg16 --runs 2 --log okdph_vgg16 \
    --omega 0.3 --beta 0.3 --gamma 1.0 --interval 1_epoch

python OKDPH.py --dataset cifar10 --model_names densenet-40-12 densenet-40-12 --runs 2 --log okdph_densenet \
    --omega 0.8 --beta 0.8 --gamma 0.5 --interval 5_batch

python OKDPH.py --dataset cifar10 --model_names wrn_20_8 wrn_20_8 --runs 2 --log okdph_wrn \
    --omega 0.8 --beta 0.8 --gamma 0.5 --interval 1_epoch

python OKDPH.py --dataset cifar10 --model_names resnet110 resnet110 --runs 2 --log okdph_resnet110 \
    --omega 0.8 --beta 0.8 --gamma 0.5 --interval 5_batch



python OKDPH.py --dataset cifar100 --model_names resnet32 resnet32 --runs 2 --log okdph_resnet32_100 \
    --omega 1.0 --beta 0.5 --gamma 1.0 --interval 5_batch

python OKDPH.py --dataset cifar100 --model_names vgg16 vgg16 --runs 2 --log okdph_vgg16_100 \
    --omega 0.3 --beta 0.3 --gamma 1.0 --interval 1_epoch

python OKDPH.py --dataset cifar100 --model_names densenet-40-12 densenet-40-12 --runs 2 --log okdph_densenet_100 \
    --omega 1.0 --beta 0.5 --gamma 0.5 --interval 5_batch

python OKDPH.py --dataset cifar100 --model_names wrn_20_8 wrn_20_8 --runs 2 --log okdph_wrn_100 \
    --omega 0.5 --beta 0.5 --gamma 0.5 --interval 5_batch

python OKDPH.py --dataset cifar100 --model_names resnet110 resnet110 --runs 2 --log okdph_resnet110_100 \
    --omega 0.8 --beta 0.8 --gamma 0.5 --interval 5_batch