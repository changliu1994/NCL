# Neighborhood Consistency Learning on Unsupervised Domain Adaptation

This repository is the official implementation for NCL on Unsupervised domain adaptation in **ACM Multimedia 2023**:
> **Rethinking Neighborhood Consistency Learning on Unsupervised Domain Adaptation [[Camera Ready](https://dl.acm.org/doi/pdf/10.1145/3581783.3612055)]** \
> Chang Liu, Lichen Wang, and Yun Fu \
> Northeastern University, Boston, MA, USA

## Environment Setup:
```bash
bash install.sh
```

## Dataset:
Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), from the official websites, and modify the path of images in each `./data/{dataset}/*.txt`.

## Training a single model

### Train model on the source domain A (s = 0)
```bash
cd code && gpu_id=0 && seed=2023 && s=0 && 
python image_source.py --gpu_id $gpu_id --seed $seed --output "ckps/s$seed" --dset office-home --max_epoch 50 --s $s
```
### Adaptation to other target domains C, P, R 
```bash
python image_NCL.py --gpu_id $gpu_id --seed $seed --da uda --dset office-home --output "ckps/t_NCL$seed" --output_src "ckps/s$seed" --s $s --k 3 --cls_par 0.1 --max_epoch 50
```

## Launch a sweep
```bash
cd code && bash run_officehome.sh 0 2023
```

## Acknowledgments
In this code we refer to the following implementations: [SHOT++](https://github.com/tim-learn/SHOT-plus/tree/master). Our code is largely built upon their wonderful implementation. 


## Reference

If our work or code helps you, please consider to cite our paper. Thank you!
```BibTeX
@inproceedings{liu2023rethinking,
  title={Rethinking Neighborhood Consistency Learning on Unsupervised Domain Adaptation},
  author={Liu, Chang and Wang, Lichen and Fu, Yun},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7247--7254},
  year={2023}
}
```