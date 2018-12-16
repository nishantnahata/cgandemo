# C-GAN Demo for image-to-image translation
This is a demo for [pix2pix Aerial-to-Map images dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) implemented [here](https://github.com/sdnr1/c-gan_pix2pix). <br>
The implementation of Condition GAN is based on a paper by Isola et al. Link : https://arxiv.org/abs/1611.07004

## Requirements
```
sudo apt-get install python3-pip build-essential libgtk2.0-dev
sudo pip3 install virtualenv

virtualenv django -p python3
source django/bin/activate

pip install tensorflow Django django-admin numpy matplotlib opencv-python
```

## Setup
### Create checkpoint directory
```
cd cgandemo
mkdir static/checkpoints
```
### Download checkpoints
Link: _
### Migrate
```
python manage.py makemigrations
python manage.py migrate
```
## Run
```
python manage.py runserver
```
*Note:* Run all commands within the created virtual environment
