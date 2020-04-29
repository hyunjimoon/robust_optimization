"""Tools to make some minimal guarantee of homogenous
data in a MongoDB collection"""

from __future__ import print_function
import pymongo
import sys

def typecheck_collection(collection, fields):
    """Assert matching types"""
    def checker(document, fields):
        for name, type_ in fields.items():
            if name in document:
                if not isinstance(document[name], type_):
                    raise TypeError('Expected {}, got {}'.format(
                        type_, type(document[name])))
    return wrap_collection(collection, checker, fields)

def convert_collection(collection, fields):
    """Ensure types by converting them"""
    def checker(document, fields):
        for name, type_ in fields.items():
            if name in document:
                document[name] = type_(document[name])
    return wrap_collection(collection, checker, fields)

def limit_collection(collection, fields):
    """Ensure only given fields are present"""
    def checker(document, fields):
        for name in document:
            if name not in fields:
                raise TypeError(
                    'Found unexpected key {}'.format(name))
    return wrap_collection(collection, checker, fields)

def checked_collection(collection, fields):
    """Ensure given fields are present"""
    def checker(document, fields):
        for name in fields:
            if name not in document:
                raise TypeError(
                    'Missing required key {}'.format(name))

    return wrap_collection(collection, checker, fields)


def wrap_collection(collection, checker, fields):
    # Don't wrap at all if optimize is on
    if sys.flags.optimize > 0:
        return collection

    updater = collection.update
    inserter = collection.insert

    def update(spec, document, **kwargs):
        checker(document, fields)
        return updater(spec, document, **kwargs)

    def insert(document, *args, **kwargs):
        checker(document, fields)
        return inserter(document, *args, **kwargs)

    collection.update = update
    collection.insert = insert

    return collection
