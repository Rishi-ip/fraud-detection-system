from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import numpy as np
import os
from .models import Transaction

# Load model once when server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'model.pkl')
model = joblib.load(MODEL_PATH)

@api_view(['POST'])
def predict(request):
    try:
        data = request.data

        # Get amount and time
        amount = float(data.get('amount', 0))
        time = float(data.get('time', 0))

        # Get V1 to V28
        features = [float(data.get(f'v{i}', 0)) for i in range(1, 29)]

        # Combine all into one list for model
        all_features = [time] + features + [amount]
        input_array = np.array(all_features).reshape(1, -1)

        # Get prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]
        confidence = round(float(max(probability)) * 100, 2)

        # Get top 3 important features
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]
        top_features = [
            {'feature': feature_names[i], 'importance': round(float(importances[i]) * 100, 2)}
            for i in top_indices
        ]

        # Save to MongoDB
        transaction = Transaction(
            amount=amount,
            time=time,
            features=features,
            is_fraud=bool(prediction),
            confidence=confidence,
            top_features=top_features
        )
        transaction.save()

        return Response({
            'is_fraud': bool(prediction),
            'confidence': confidence,
            'top_features': top_features,
            'message': '🚨 FRAUD DETECTED' if prediction == 1 else '✅ Transaction is SAFE'
        })

    except Exception as e:
        return Response({'error': str(e)}, status=400)


@api_view(['GET'])
def history(request):
    transactions = Transaction.objects.all()[:20]
    data = []
    for t in transactions:
        data.append({
            'amount': t.amount,
            'is_fraud': t.is_fraud,
            'confidence': t.confidence,
            'top_features': t.top_features,
            'created_at': t.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    return Response(data)


def dashboard(request):
    return render(request, 'fraud/dashboard.html')


def home(request):
    return render(request, 'fraud/home.html')