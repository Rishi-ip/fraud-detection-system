import mongoengine as me
from datetime import datetime

class Transaction(me.Document):
    amount = me.FloatField(required=True)
    time = me.FloatField(required=True)
    
    features = me.ListField(me.FloatField())
    
    is_fraud = me.BooleanField(default=False)
    confidence = me.FloatField(default=0.0)
    top_features = me.ListField()
    
    created_at = me.DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'transactions',
        'ordering': ['-created_at']
    }
    
    def __str__(self):
        status = "FRAUD" if self.is_fraud else "SAFE"
        return f"Transaction {status} - Amount: {self.amount}"