from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langdetect import detect
import speech_recognition as sr
import io
# Create your views here.

def detect_en(text):
    if detect(text) == "en":
        return True
    else:
        return False

@csrf_exempt
def speech(request):
    if request.method == "POST":
        print(request.body)
        print(request.FILES)
        nchannels = 2
        sampwidth = 2
        framerate = 8000
        nframes = 100

        import wave

        name = 'output.wav'
        audio = wave.open(name, 'wb')
        audio.setnchannels(nchannels)
        audio.setsampwidth(sampwidth)
        audio.setframerate(framerate)
        audio.setnframes(nframes)

        blob = request.body.read() # such as `blob.read()`
        audio.writeframes(blob)

        r = sr.Recognizer()
        with open("audio.wav", 'wb+' ) as destination:
            for chunk in request.FILES['File'].chunks():
                destination.write(chunk)
        # file_obj = io.BytesIO()  # create file-object
        # file_obj.write(request.FILES['File'].read()) # write in file-object
        # file_obj.seek(0) # move to beginning so it will read from beginning
        with sr.AudioFile("audio.wav") as source:
    # #     # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print(text)
            if detect_en(text) == True:
                return JsonResponse({'lang':"English"})
            else:
                return JsonResponse({'lang':'Non English'})
    # return render(request, 'input/front_page.html')


@csrf_exempt
def text(request):
    if request.method == "POST":
        if detect_en(str(request.body)) == True:
            return JsonResponse({'lang':"English"})
        else:
            return JsonResponse({'lang':'Non English'})
    else:
        return render(request, 'input/text_input.html')

@csrf_exempt
def file(request):
    if request.method == "POST":
        print(request.body)
        print(request.FILES['File'])
        str_file = ""
        for chunk in request.FILES['File'].chunks():
            str_file += str(chunk)
        if detect_en(str_file) == True:
            return JsonResponse({'lang':"English"})
        else:
            return JsonResponse({'lang':'Non English'})