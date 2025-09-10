from decimal import Decimal
from datetime import date, datetime, timedelta
import json

class DBEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        return super(DBEncoder, self).default(obj)