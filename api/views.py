from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from . import temp
# Create your views here.


class ChatBot(APIView):
    def post(self,request):
        question=request.data['question']
        
        if(temp.is_law_related(question)==False):
            return Response({"message":str(temp.trainer(question))})
            
        else:
            answer=temp.get_answer(question)
            return Response({"message":answer},status=200)