## Run
Get the coordinates of each grid point in the whole plane.

```bash
# resnet32
python grid_loss.py --dataset cifar10 --model resnet32 --method Base --x_tuple -25 30 --y_tuple -25 35 --save resnet32/cifar10_Base

python grid_loss.py --dataset cifar10 --model resnet32 --method dml --x_tuple -25 25 --y_tuple -20 30 --save resnet32/cifar10_dml

python grid_loss.py --dataset cifar10 --model resnet32 --method KDCL --x_tuple -15 30 --y_tuple -25 25 --save resnet32/cifar10_KDCL


python grid_loss.py --dataset cifar100 --model resnet32 --method Base --x_tuple -40 40 --y_tuple -30 50 --save resnet32/cifar100_Base

python grid_loss.py --dataset cifar100 --model resnet32 --method dml --x_tuple -35 45 --y_tuple -40 45 --save resnet32/cifar100_dml

python grid_loss.py --dataset cifar100 --model resnet32 --method KDCL --x_tuple -25 45 --y_tuple -35 35 --save resnet32/cifar100_KDCL



# densenet
python grid_loss.py --dataset cifar10 --model densenet-40-12 --method Base --x_tuple -25 30 --y_tuple -25 30 --save densenet/cifar10_Base --gid 2

python grid_loss.py --dataset cifar10 --model densenet-40-12 --method dml --x_tuple -20 25 --y_tuple -20 25 --save densenet/cifar10_dml --gid 1

python grid_loss.py --dataset cifar10 --model densenet-40-12 --method KDCL --x_tuple -15 25 --y_tuple -20 20 --save densenet/cifar10_KDCL --gid 0


python grid_loss.py --dataset cifar100 --model densenet-40-12 --method Base --x_tuple -40 45 --y_tuple -30 40 --save densenet/cifar100_Base

python grid_loss.py --dataset cifar100 --model densenet-40-12 --method dml --x_tuple -35 40 --y_tuple -25 35 --save densenet/cifar100_dml --gid 0

python grid_loss.py --dataset cifar100 --model densenet-40-12 --method KDCL --x_tuple -25 35 --y_tuple -25 30 --save densenet/cifar100_KDCL


# wrn
python grid_loss.py --dataset cifar10 --model wrn_20_8 --method Base --x_tuple -30 30 --y_tuple -15 35  --save wrn/cifar10_Base --pca_end 120 --gid 2

python grid_loss.py --dataset cifar10 --model wrn_20_8 --method dml --x_tuple -25 25 --y_tuple -15 35  --save wrn/cifar10_dml --pca_end 120 --gid 2

python grid_loss.py --dataset cifar10 --model wrn_20_8 --method KDCL --x_tuple -20 20 --y_tuple -15 40  --save wrn/cifar10_KDCL --pca_end 120 --gid 0


python grid_loss.py --dataset cifar100 --model wrn_20_8 --method Base --x_tuple -45 45 --y_tuple -20 25  --save wrn/cifar100_Base --pca_end 120

python grid_loss.py --dataset cifar100 --model wrn_20_8 --method dml --x_tuple -40 40 --y_tuple -20 30  --save wrn/cifar100_dml --pca_end 120

python grid_loss.py --dataset cifar100 --model wrn_20_8 --method KDCL --x_tuple -30 30 --y_tuple -10 25  --save wrn/cifar100_KDCL --pca_end 120


# vgg16
python grid_loss.py --dataset cifar10 --model vgg16 --method Base --x_tuple -30 35 --y_tuple -25 30  --save vgg16/cifar10_Base --pca_end 240

python grid_loss.py --dataset cifar10 --model vgg16 --method dml --x_tuple -30 35 --y_tuple -25 25  --save vgg16/cifar10_dml --pca_end 240

python grid_loss.py --dataset cifar10 --model vgg16 --method KDCL --x_tuple -20 20 --y_tuple -25 25 --save vgg16/cifar10_KDCL --pca_end 240



python grid_loss.py --dataset cifar100 --model vgg16 --method Base --x_tuple -50 50 --y_tuple -30 40  --save vgg16/cifar100_Base

python grid_loss.py --dataset cifar100 --model vgg16 --method dml --x_tuple -45 45 --y_tuple -30 40  --save vgg16/cifar100_dml

python grid_loss.py --dataset cifar100 --model vgg16 --method KDCL --x_tuple -35 35 --y_tuple -25 35 --save vgg16/cifar100_KDCL


# resnet110
python grid_loss.py --dataset cifar10 --model resnet110 --method Base --x_tuple -10 10 41 --y_tuple -25 25 51 --save resnet110/cifar10_Base --pca_end 180 --pca_start 60

python grid_loss.py --dataset cifar10 --model resnet110 --method dml --x_tuple -10 10 41 --y_tuple -25 35 61 --save resnet110/cifar10_dml --pca_end 180 --pca_start 60

python grid_loss.py --dataset cifar10 --model resnet110 --method KDCL --x_tuple -10 10 41 --y_tuple -20 20 41  --save resnet110/cifar10_KDCL --pca_end 180 --pca_start 60


python grid_loss.py --dataset cifar100 --model resnet110 --method Base --x_tuple -35 40 --y_tuple -45 45  --save resnet110/cifar100_Base --pca_end 240

python grid_loss.py --dataset cifar100 --model resnet110 --method dml --x_tuple -30 35 --y_tuple -40 40  --save resnet110/cifar100_dml --pca_end 240

python grid_loss.py --dataset cifar100 --model resnet110 --method KDCL --x_tuple -25 45 --y_tuple -35 35  --save resnet110/cifar100_KDCL --pca_end 180
```