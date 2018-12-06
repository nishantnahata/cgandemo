# C-GAN Demo for image-to-image translation

## Requirements
```
sudo apt-get install python3-pip
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
