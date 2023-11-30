from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.template import RequestContext

def about3(request):
    return render(request,'about3.html')

