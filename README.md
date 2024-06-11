# ACAPTCHA

ACHAPTCHA is composed of two components: (1) an adversarial text captcha generation method, and (2) an adversarial text captcha attack method.



### Prerequisites

* Install necessary packages from requirements.txt
* Download crnn.rar from [here](https://drive.google.com/file/d/1NFYGkOPUycYjqvOBRrwsQArRWCBSI0DK/view?usp=sharing) and unzip them into `crnn`



### Getting Started

**1. Generate host captchas including different security features** 

Modify the python file into `gen_v*.py`, where `*` represents the version number(i.e., 1~6).

```
cd design
python gen_v1.py
```



**2. Add adversarial example on HCAPTCHA**

Firstly, you need to train attack models. Data will be stored in `lmdb`.

```
cd crnn
python tool/create_dataset.py --out lmdb/version/v1_1/train --folder ../datasets/version/v1_1/train
python tool/create_dataset.py --out lmdb/version/v1_1/val --folder ../datasets/version/v1_1/val
python train.py --trainroot lmdb/version/v1_1/train --valroot lmdb/version/v1_1/val
```

Then, you can generate adversarial examples using different adversarial example algorithms (`fgsm.py`, `i-fgsm.py`, or `deepfool.py`).

```
python fgsm.py
```



**3.  Eliminate perturbation of ACAPTCHA through image processing techniques**

We integrate 9 image processing techniques (Pix2Pix, total variation loss, three data compression algorithms, and four image filters) to remove adversarial perturbation in ACAPTCHAs.

```
python filter.py
```



**4. Solve the processed captchas using attack models**

```
python test.py
```

