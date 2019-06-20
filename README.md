Encoding Word Order in Complex-valued Embedding
=====================

This repository contains code which reproduces experiments in the paper [Encoding Word Order in Complex-valued Embedding]<!-- (https://arxiv.org/abs/1705.09792). -->

Requirements
------------

Install requirements for computer vision experiments with pip:

```
pip install numpy tensorflow keras scipy pandas nltk pickle 
```


Depending on your Python installation you might want to use anaconda or other tools.


Experiments
-----------

Run models:

    ```
    cd Fasttext/CNN/BiLSTM
    python train.py
    ```

This model is for mr and subj. The default is for mr, and if you want to run this model on subj, you can:

    ```
    cd Fasttext/CNN/BiLSTM
    python train.py --data subj
    ```

You can use other classification dataset. Please put your dataset on dir data, and preprocess your data according to mr.




Citation
--------
<!-- 
Please cite our work as 

```
@ARTICLE {,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
```
 -->