
https://github.com/ynes99/BraTS_Segmentation/assets/72978456/1a71effa-80ce-4006-bc72-e14819b26247

https://github.com/ynes99/BraTS_Segmentation/assets/72978456/550060d4-e6a9-4d9e-a5aa-bd88b9478a15
 ---

<div align="center">    
 
# Brain Tumor segmentation (delineation) of Gliomas using Transformers


<!--  
Conference   
-->   
</div>
 
## Description   
We delineate the borders of the brain tumors gliomas using different Transformer-based Deep Learning Models.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ynes99/BraTS_Segmentation

# install project   
cd BraTS_Segmentation
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, The Jupiter Notebook [dataset exploration](https://github.com/ynes99/BraTS_Segmentation/blob/main/Dataset%20exploration.ipynb) introduces us to the dataset used in this project which is the BRATS dataset.
 In The Notebook [Segment_Anything_Model](https://github.com/ynes99/BraTS_Segmentation/blob/main/Segment_Anything_Model.ipynb) has the necessary preprocessing for the data to be inputted directly into the model.
 [x] SAM 
 [x] Trans-U-Net (Progress)
 [ ] Swin-U-Net
### Presentation of last studies internship   

[Presentation end of studies internship.pdf](https://github.com/ynes99/BraTS_Segmentation/files/12608422/Presentation.end.of.studies.internship.pdf)



### Citation   
```
@misc{Ines,
  title={Delineation of Brain Tumors for CAD},
  author={Ines Bouhelal},
  year={2023}
}
```   
