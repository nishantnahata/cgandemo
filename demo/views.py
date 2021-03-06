from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.views.generic.base import View
from .c_gan import C_GAN
from cgandemo.settings import MEDIA_ROOT
from django.core.files.storage import FileSystemStorage
import cv2
import os

class Demo(View):
    fs = FileSystemStorage()
    template = 'demo/index.html'
    c_gan = C_GAN()

    def get(self, request):
        return render(request, self.template)

    def post(self, request):
        try:
            image = request.FILES['image']
            filename = self.fs.save(image.name, image)
            url = os.path.join(MEDIA_ROOT, filename)
            input_image, output_image = self.c_gan.predict(cv2.imread(url))
            self.fs.delete('output_images/' + filename)
            cv2.imwrite(os.path.join(MEDIA_ROOT,
                                     'output_images/' + filename),
                        output_image)
            self.fs.delete('input_images/' + filename)
            cv2.imwrite(os.path.join(MEDIA_ROOT,
                                     'input_images/' + filename),
                        input_image)
            self.fs.delete(filename)
            return render(request, self.template,
                          {'output': self.fs.url('output_images/' + filename),
                           'input': self.fs.url('input_images/' + filename)})
        except:
            return render(request, self.template,
                          {'error': "Please check the file you uploaded!"})
