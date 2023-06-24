# XDA

Official code for our paper "Taming Prompt-based Data Augmentation for Extreme Multi-Label Text Classification”.  

# Requirements: 
* python==3.7.10
* numpy==1.21.2
* scipy==1.7.2
* scikit-learn==0.22
* torch==1.10.0
* gensim==4.1.2
* nltk==3.6.5
* tqdm==4.62.3 
* tokenizers==0.7.0 
* transformers==2.11.0


## Datasets
XDA uses the same dataset with [DEPL](https://aclanthology.org/2023.findings-eacl.81.pdf).

please download them from the following links.
* [EUR-Lex](https://drive.google.com/file/d/1q7a2NkZh-1vEyCchXdDKhRttjkBoiTHr/view?usp=drive_link)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)




## Experiments
train and eval
```bash
bash script/run_xda.sh <dataset> <gpu_id> 
```
