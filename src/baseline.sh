# DML
python 01_DML.py --dataset cifar10 --model_names resnet32 resnet32 --runs 5 --log dml_resnet32
python 01_DML.py --dataset cifar10 --model_names vgg16 vgg16 --runs 5 --log dml_vgg16
python 01_DML.py --dataset cifar10 --model_names densenet-40-12 densenet-40-12 --runs 5 --log dml_densenet
python 01_DML.py --dataset cifar10 --model_names wrn_20_8 wrn_20_8 --runs 5 --log dml_wrn
python 01_DML.py --dataset cifar10 --model_names resnet110 resnet110 --runs 5 --log dml_resnet110

python 01_DML.py --dataset cifar100 --model_names resnet32 resnet32 --runs 5 --log dml_resnet32
python 01_DML.py --dataset cifar100 --model_names vgg16 vgg16 --runs 5 --log dml_vgg16
python 01_DML.py --dataset cifar100 --model_names densenet-40-12 densenet-40-12 --runs 5 --log dml_densenet
python 01_DML.py --dataset cifar100 --model_names wrn_20_8 wrn_20_8 --runs 5 --log dml_wrn
python 01_DML.py --dataset cifar100 --model_names resnet110 resnet110 --runs 5 --log dml_resnet110


# ONE
python 02_ONE.py --dataset cifar10 --model_names resnet32_one --num_branches 2 --runs 5 --log one_resnet32
python 02_ONE.py --dataset cifar10 --model_names vgg16_one --num_branches 2 --runs 5 --log one_vgg16
python 02_ONE.py --dataset cifar10 --model_names densenet-40-12_one --num_branches 2 --runs 5 --log one_densenet
python 02_ONE.py --dataset cifar10 --model_names wrn_20_8_one --num_branches 2 --runs 5 --log one_wrn
python 02_ONE.py --dataset cifar10 --model_names resnet110_one --num_branches 2 --runs 5 --log one_resnet110

python 02_ONE.py --dataset cifar100 --model_names resnet32_one --num_branches 2 --runs 5 --log one_resnet32
python 02_ONE.py --dataset cifar100 --model_names vgg16_one --num_branches 2 --runs 5 --log one_vgg16
python 02_ONE.py --dataset cifar100 --model_names densenet-40-12_one --num_branches 2 --runs 5 --log one_densenet
python 02_ONE.py --dataset cifar100 --model_names wrn_20_8_one --num_branches 2 --runs 5 --log one_wrn
python 02_ONE.py --dataset cifar100 --model_names resnet110_one --num_branches 2 --runs 5 --log one_resnet110


# KDCL
python 03_KDCL.py --dataset cifar10 --model_names resnet32 resnet32 --runs 5 --log KDCL_resnet32
python 03_KDCL.py --dataset cifar10 --model_names vgg16 vgg16 --runs 5 --log KDCL_vgg16
python 03_KDCL.py --dataset cifar10 --model_names densenet-40-12 densenet-40-12 --runs 5 --log KDCL_densenet
python 03_KDCL.py --dataset cifar10 --model_names wrn_20_8 wrn_20_8 --runs 5 --log KDCL_wrn
python 03_KDCL.py --dataset cifar10 --model_names resnet110 resnet110 --runs 5 --log KDCL_resnet110

python 03_KDCL.py --dataset cifar100 --model_names resnet32 resnet32 --runs 5 --log KDCL_resnet32
python 03_KDCL.py --dataset cifar100 --model_names vgg16 vgg16 --runs 5 --log KDCL_vgg16
python 03_KDCL.py --dataset cifar100 --model_names densenet-40-12 densenet-40-12 --runs 5 --log KDCL_densenet
python 03_KDCL.py --dataset cifar100 --model_names wrn_20_8 wrn_20_8 --runs 5 --log KDCL_wrn
python 03_KDCL.py --dataset cifar100 --model_names resnet110 resnet110 --runs 5 --log KDCL_resnet110



# OKDDIP
python 04_OKDDip.py --dataset cifar100 --model_names resnet32_okddip --num_branches 2 --runs 5 --log OKDDip_resnet32
python 04_OKDDip.py --dataset cifar100 --model_names vgg16_okddip --num_branches 2 --runs 5 --log OKDDip_vgg16
python 04_OKDDip.py --dataset cifar100 --model_names densenet-40-12_okddip --num_branches 2 --runs 5 --log OKDDip_densenet
python 04_OKDDip.py --dataset cifar100 --model_names wrn_20_8_okddip --num_branches 2 --runs 5 --log OKDDip_wrn
python 04_OKDDip.py --dataset cifar100 --model_names resnet110_okddip --num_branches 2 --runs 5 --log OKDDip_resnet110


# FFL
python 05_FFL.py --dataset cifar100 --model_names resnet32_ffl resnet_fm --num_branches 2 --runs 5 --log ffl_resnet32
python 05_FFL.py --dataset cifar100 --model_names vgg16_ffl vgg_fm --num_branches 2 --runs 5 --log ffl_vgg16
python 05_FFL.py --dataset cifar100 --model_names densenet-40-12_ffl densenet_fm --num_branches 2 --runs 5 --log ffl_densenet2
python 05_FFL.py --dataset cifar100 --model_names wrn_20_8_ffl resnet_fm --num_branches 2 --runs 5 --log ffl_wrn
python 05_FFL.py --dataset cifar100 --model_names resnet110_ffl resnet_fm --num_branches 2 --runs 5 --log ffl_resnet110


# PCL
python 06_PCL.py --dataset cifar100 --model_names resnet32_pcl resnet32_pcl --num_branches 2 --runs 5 --log pcl_resnet32
python 06_PCL.py --dataset cifar100 --model_names vgg16_pcl vgg16_pcl --num_branches 2 --runs 5 --log pcl_vgg16
python 06_PCL.py --dataset cifar100 --model_names densenet-40-12_pcl densenet-40-12_pcl --runs 5 --log pcl_densenet-40-12
python 06_PCL.py --dataset cifar100 --model_names wrn_20_8_pcl wrn_20_8_pcl --num_branches 2 --runs 5 --log pcl_wrn
python 06_PCL.py --dataset cifar100 --model_names resnet110_pcl resnet110_pcl --num_branches 2 --runs 5 --log pcl_resnet110


# EMA
python 11_EMA.py --dataset cifar10 --model_names densenet-40-12 densenet-40-12 --runs 5 --log EMA_densenet
python 11_EMA.py --dataset cifar10 --model_names wrn_20_8 wrn_20_8 --runs 5 --log EMA_wrn
python 11_EMA.py --dataset cifar10 --model_names resnet110 resnet110 --runs 5 --log EMA_resnet110
python 11_EMA.py --dataset cifar10 --model_names resnet32 resnet32 --runs 5 --log EMA_resnet32
python 11_EMA.py --dataset cifar10 --model_names vgg16 vgg16 --runs 5 --log EMA_vgg16


# SWA
python 12_SWA.py --dataset cifar10 --model_names densenet-40-12 --runs 5 --log SWA_densenet
python 12_SWA.py --dataset cifar10 --model_names wrn_20_8 --runs 5 --log SWA_wrn
python 12_SWA.py --dataset cifar10 --model_names resnet110 --runs 5 --log SWA_resnet110
python 12_SWA.py --dataset cifar10 --model_names resnet32 --runs 5 --log SWA_resnet32
python 12_SWA.py --dataset cifar10 --model_names vgg16 --runs 5 --log SWA_vgg16


# SAM
python 13_SAM.py --dataset cifar100 --model_names densenet-40-12 --log SAM_densenet --runs 5
python 13_SAM.py --dataset cifar100 --model_names wrn_20_8 --log SAM_wrn --runs 5
python 13_SAM.py --dataset cifar100 --model_names resnet110 --log SAM_resnet110 --runs 5
python 13_SAM.py --dataset cifar100 --model_names resnet32 --log SAM_resnet32 --runs 5
python 13_SAM.py --dataset cifar100 --model_names vgg16 --log SAM_vgg16 --runs 5


# KR
python 14_KR_adapt.py --dataset cifar100 --model_names resnet32 --log KR_resnet32 --runs 5
python 14_KR_adapt.py --dataset cifar100 --model_names vgg16 --log KR_vgg16 --runs 5
python 14_KR_adapt.py --dataset cifar100 --model_names densenet-40-12 --log KR_densenet --runs 5
python 14_KR_adapt.py --dataset cifar100 --model_names wrn_20_8 --log KR_wrn --runs 5
python 14_KR_adapt.py --dataset cifar100 --model_names resnet110 --log KR_resnet110 --runs 5
