import time
from threading import Thread


class ThreadManager:
    def __init__(self):
        self.tasks = []
        self.stop = None

    def main_loop(self):
        for task in self.tasks:
            name, task_method, on_complete_method, timer = task
            time.sleep(timer)
            out = task_method()
            if on_complete_method is not None:
                if out is not None:
                    on_complete_method(*out)
                else:
                    on_complete_method()
        self.tasks.clear()

    def add_task(self, name, task, on_complete=None, timer=0):
        for task_name, _, __, ___ in self.tasks:
            if task_name == name:
                return
        self.tasks.append((name, task, on_complete, timer))

        self.start_loop()

    def start_loop(self):
        self.stop = False
        Thread(target=self.main_loop).start()

    def stop_loop(self):
        self.stop = True
