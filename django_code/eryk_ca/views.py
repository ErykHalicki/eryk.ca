from django.shortcuts import render

def timeline_view(request):
    return render(request, 'index.html')
def about_me(request):
    return render(request, 'about_me.html')

# Create your views here.


