# MAC-Multi-scale-Attention-Cascade
This is an official code for "MAC: Mutil-scale Attention Cascade for aerial image segmentation", which is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) open source toolbox of semantic segmentation. This work achieves 69.06 mIoU on iSAID dataset and 73.37 on ISPRS Vaihingen dataset. Accepted by ICPRAM2024, Oral  
![Visualziation Sample](https://github.com/EricBooob/Multi-scale-Attention-Cascade-for-Aerial-Image-Segmentation/blob/main/visualization%20compare.png)   
![Comparison Study](https://github.com/EricBooob/Multi-scale-Attention-Cascade-for-Aerial-Image-Segmentation/blob/main/comparison%20study%20on%20ISAID.png)  


## Content
1.Requirements      
2.Dataset  
3.Main code  
4.Testing      
5.Acknowledge  
6.Reference  

## !!!IMPORTANT!!! Before using, please read and be aware of the ```SoftwareLicenseAgreement_20230807_v1.pdf```   

## Requirements  
 
We have tested our code with 

```
python=3.10.0  
pytorch=1.12.1   
CUDA=11.4
CuDNN=8.3.2
mmcv-full=1.6.2
```   

To install [mmcv](https://github.com/open-mmlab/mmcv) and other related prerequisites, please follow the procedure provided in  

```MAC-mmsegmentation/docs/en/get_started.md```

## Dataset
We utlized [iSAID](https://captain-whu.github.io/iSAID/) dataset for benckmark.  

### iSAID dataset 
Original Input images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)  
Data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)
The detailed dataset preparation procedure is provided in  

``` MAC-mmsegmentation/docs/en/dataset_prepare.md``` 

If you you have downloaded the dataset, please store it at the following path:  

``` MAC-mmsegmentation/data/iSAID ```  

## Main Code
The MAC_head.py contains the main part of MAC model, you could find it at:  

``` MAC-mmsegmentation/mmseg/models/decode_heads/MAC_heads.py ```   


## Testing 
1. we provide our pretrained model checkpoint ``` mac_latest.pth ```   here:  
[Google Drive](https://drive.google.com/file/d/1mIoe6xK50T65qWHZMd4_Yrvy7rf2itIi/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1yi6tJWVgKfI1hKyiVYoGRA) (PSW:nn1t)  
3. After downloading the checkpoint, please store it at the following path:

``` MAC-mmsegmentation/work_dirs/mac_isaid/mac_latest.pth ```   

3. The configuration file is stored at the following path:  

``` MAC-mmsegmentation/configs/_base_/models/mac_isaid.py ``` and  ``` MAC-mmsegmentation/configs/mac/MAC_swin_isaid.py ```   

4. To test the performance of provide checkpoint, please run the following command line:  

```
cd MAC-mmsegmentation
python tools/test.py configs/mac/MAC_swin_isaid.py work_dirs/mac_isaid/mac_latest.pth --eval mIoU mFscore  --show-dir work_dirs/mac_isaid/outs/
```

5. The visulization resuls will be stored at ``` MAC-mmsegmentation/work_dirs/mac_isaid/outs/ ```


## Acknowledgment
* This model is based on the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), thanks to the contributors to the project.  

## Reference    

```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
