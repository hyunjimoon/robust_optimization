"""Log handler and formatter to log messages to MongoDB

Not (yet) guaranteed to work with arbitrary extras fields"""

import uuid
import traceback
import logging

from datetime import datetime
from socket import gethostname

class JsonFormatter(logging.Formatter):
    def __init__(self, **kwargs):
        self.host = gethostname()
        self.extra = kwargs

    def format(self, record):
        # We want a dict and not a LogRecord object.
        data = record.__dict__.copy()

        if record.args:
            msg = record.msg % record.args
        else:
            msg = record.msg

        args = record.args if isinstance(record.args, tuple) else (record.args,)

        data.update(
            time=datetime.utcnow(),
            host=self.host,
            message=msg,
            args=tuple(unicode(arg) for arg in args))
        data.update(self.extra)
        if data.get('exc_info'):
            data['exc_info'] = self.formatException(data['exc_info'])

        return data


class MongoHandler(logging.Handler):
    """Meant to be used with the MongoFormatter, but any formatter that gives
    records that MongoDB will accept should work.
    Mandatory first argument is a collection object from MongoDB"""
    def __init__(self, collection, formatter=None, level=logging.NOTSET):
        logging.Handler.__init__(self, level)
        self.collection = collection
        if formatter == None:
            formatter = JsonFormatter()
        self.formatter = formatter

    def emit(self, record):
        self.collection.insert(self.format(record))
