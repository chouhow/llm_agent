from decimal import Decimal
from datetime import date, datetime
import json

class DBEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super(DBEncoder, self).default(obj)