from time import time


class Timer(object):
    def __init__(self, start: int = 0, indentation: int = 0):
        self.start = start or time()
        self.indentation = indentation

    def __enter__(self):
        print('#%s enter at %.2f seconds' % ('\t' * self.indentation, time() - self.start))
        return self.tab()

    def track(self, message: str):
        print('#%s %s -> at time %.2f' % ('\t' * self.indentation, message, time() - self.start))

    def tab(self):
        return Timer(start=self.start, indentation=self.indentation+1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('#%s exit at %.2f seconds' % ('\t' * self.indentation, time() - self.start))
        return False
