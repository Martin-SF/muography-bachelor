# %%


from time import time as time_time
from time import sleep
class stopwatch:
    def __init__(self, start = 'just_initialize'):
        if (start == 'just_initialize'):
            self.not_started_warning = True
        self.start_time = time_time()

    def start(self):
        self.start_time = time_time()
        self.not_started_warning = False
    
    def stop(self):
        if (self.not_started_warning):
            print('Warning: stopwatch assumes start_time at object creation')
            self.not_started_warning = False
        elapsed_time_in_ms = (time_time()-self.start_time)*1000
        print(f'stopwatch measured {(elapsed_time_in_ms):.1f}ms')
        return elapsed_time_in_ms

t = stopwatch()
t.start()
sleep(2)
t.stop()
