"""Locking using MongoDB

Not meant to be extremely fast, but a convenient way to get distributed
locks if you have MongoDB in your stack"""
import contextlib
import datetime
import socket

HOSTNAME = socket.gethostname()


class LockException(Exception):
    def __init__(self, message, lock_document=None):
        super(LockException, self).__init__(message)
        self.lock_document = lock_document

class Locked(LockException):
    pass

class ConcurrentLockModificationError(LockException):
    pass


def engage_lock(lock_collection, lock_id, lock_subdoc):
    """Direct access to locking mechanism, you probably should stick with
    Lock.lock or the context manager 'lock' instead"""
    document = lock_collection.find_one({'_id': lock_id})
    if not document:
        try:
            lock_collection.insert(
                {'_id': lock_id,
                 lock_subdoc: {'locked': False,
                             'updated': datetime.datetime.utcnow(),
                             'version': 1}})
            document = lock_collection.find_one({'_id': lock_id})
        except KeyError:
            pass  # Concurrent creation
    if not document.get(lock_subdoc):
        try:
            lock_collection.update(
                {'_id': lock_id},
                {lock_subdoc: {'locked': False,
                             'updated': datetime.datetime.utcnow(),
                             'version': 1}})
            document = lock_collection.find_one({'_id': lock_id})
        except KeyError:
            pass

    if document.get(lock_subdoc).get('locked'):
        raise Locked('Lock is already locked', document)

    write_result = lock_collection.update({
        '_id': lock_id,
        lock_subdoc + '.version': document[lock_subdoc]['version'],
        lock_subdoc + '.locked': document[lock_subdoc]['locked']
    }, {
        '$set': {lock_subdoc + '.locked': True,
                 lock_subdoc + '.updated': datetime.datetime.utcnow(),
                 lock_subdoc + '.host': HOSTNAME},
        '$inc': {lock_subdoc + '.version': 1}
    })

    if write_result.get('n') != 1:
        raise Locked(lock_collection.find_one({'_id': lock_id}))
    return document

def release_lock(lock_collection, lock_id, lock_subdoc, document):
    """Direct access to unlocking mechanism, you probably should stick with
    Lock.unlock or the context manager 'lock' instead"""

    write_result = lock_collection.update(
        {
            '_id': lock_id,
            lock_subdoc + '.version': document[lock_subdoc]['version'] + 1,
            lock_subdoc + '.locked': True
        },
        {
            '$set': {lock_subdoc + '.locked': False,
                     lock_subdoc + '.updated': datetime.datetime.utcnow()},
            '$inc': {lock_subdoc + '.version': 1}
        }
    )
    if write_result.get('n') != 1:
        raise ConcurrentLockModificationError(
            lock_collection.find_one({'_id': lock_id}))


class Lock(object):
    """Lock as an object"""
    def __init__(self, lock_collection, lock_id, lock_subdoc=None):
        self._lock_collection = lock_collection
        self._lock_id = lock_id
        self._lock_subdoc = lock_subdoc
        self._document = None
        self.is_locked = False  # Informational only!

    def lock(self):
        self._document = engage_lock(self._lock_collection,
                                     self._lock_id, self._lock_subdoc)
        self.is_locked = True

    def unlock(self):
        if self._document is None:
            raise LockException('Not locked')
        release_lock(self._lock_collection, self._lock_id, self._lock_subdoc,
                self._document)
        self.is_locked = False

    def __repr__(self):
        return 'Lock {}[{}].{} status {}'.format(self._lock_collection,
                                                 self._lock_id,
                                                 self._lock_subdoc,
                                                 self.is_locked)

@contextlib.contextmanager
def lock(lock_collection, lock_id, lock_subdoc):
    """Lock as a context manager"""
    engage_lock(lock_collection, lock_id, lock_subdoc)
    try:
        yield  # This is where the calling code runs
    finally:
        release_lock(lock_collection, lock_id, lock_subdoc)
