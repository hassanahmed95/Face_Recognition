# importing all necessary packages. . . .
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from passporteye import read_mrz
import numpy as np
import cv2
import pdb
import json
from .face_recog import main


@csrf_exempt
def detect(request):
    if request.method == "POST":
        X = request.POST.get('image')
        X = json.loads(X)
        if request.POST.get("image", None) is not None:
            # pdb.set_trace()
            images = _grab_image(stream=X["images"])
            response = main(images)
            return JsonResponse({response:"."})


def _grab_image(stream=None):
    data = []
    if stream is not None:
        for image in stream:
            face  = cv2.imread(image)
            data.append(face)
        return data
