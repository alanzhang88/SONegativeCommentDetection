from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .Classify import Classify
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def classify(request):
    # return render(request, 'demo.html', {'labels': labels})
    # if request.is_ajax():
	   #  if request.method == 'POST':
	   #      print('Raw Data: "%s"' % request.body)
    # return JsonResponse({'body': request.body})
	body_unicode = request.body.decode('utf-8')
	body = json.loads(body_unicode)
	comments = body['comments']
	print(comments)
	lstm_labels, cnn_labels, fasttext_labels= Classify.predict(comments)
	return JsonResponse({'lstm_labels': lstm_labels, 'cnn_labels': cnn_labels, 'fasttext_labels': fasttext_labels})

def helloworld(request):
	return HttpResponse('Hello World')
