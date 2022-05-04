# %%
from time import time as time_s
from time import time_ns as time_ns
from time import perf_counter as timer_perf
from prettytable import PrettyTable
# import pprint

class stopwatch:

    time_units_dict = {'µs': int(1e6), 'ms': int(1e3), 's': 1, 'min': 1/60}
    
    def __init__(self, start = False, task_name = None,
                print_task = False, mode = 'table',
                title = None,
                return_results = False,
                print_all_tasks = False,
                time_unit = 'ms',
                min_task_name_length = 10,
                min_time_length = 7,
                selfexecutiontime_micros = 0):
        self.mode = mode
        self.title = title
        self.return_results = return_results
        self.print_all_tasks = print_all_tasks
        self.time_unit = time_unit
        self.min_task_name_length = min_task_name_length
        self.min_time_length = min_time_length
        self.selfexecutiontime = selfexecutiontime_micros / int(1e6)
        self.__stopwatch_running = False
        if (start):
            self.task(task_name, print_task)

    # def start(self, task_name = None, print_task = False, **kwargs):
    #     self.__dict__.update(kwargs)
    #     self.task(task_name, print_task)

    def settings(self, **kwargs):
        self.__dict__.update(kwargs)

    def task(self, task_name = None, print_task = False):
        if (not self.__stopwatch_running):
            self.timecodes = []*100
            self.tasks = []*100
            self.durations = []*100
            self.print_task_bool = [bool]*100
            self.task_durations = {}
            self.__stopwatch_running = True

        if (task_name == None):
            task_name = f'task #{len(self.timecodes)+1}'
        self.__save_timecode(task_name)
        self.print_task_bool.append((print_task or self.print_all_tasks))

        if (len(self.timecodes)>1):
            self.__save_duration()

            if (self.print_task_bool[-2]):
                self.__print_output(True)
            if (self.return_results):
                return self.task_durations

    def stop(self, silent = False, task_name = 'TOTAL', ):
        if (not self.__stopwatch_running):
            raise ValueError('Error: stopwatch has to be started!')
        self.task(task_name)
        self.__save_duration(0, -1)
        self.__stopwatch_running = False
        if (not silent):
            self.__print_output()
        
        if (self.return_results):
            return self.task_durations

    # def s(self, *args, **kwargs):
    #     self.start(*args, **kwargs)

    def p(self, *args, **kwargs):
        self.stop(*args, **kwargs)

    def t(self, *args, **kwargs):
        self.task(*args, **kwargs)

    # private functions
    def __save_timecode(self, task_name):
        # self.timecodes.append(time_s())
        # self.timecodes.append(time_ns() / int(1e9))
        self.timecodes.append(timer_perf())
        self.tasks.append(task_name)
    
    def __save_duration(self, first = -2, second = -1):
        if (first == 0 and second == -1):
            elapsed_time = sum(self.durations)
        else:
            elapsed_time = self.__f_elapsed_time(self.timecodes[first], self.timecodes[second])
        self.durations.append(elapsed_time)
        if (self.return_results):
            self.task_durations[self.tasks[len(self.durations)-1]] = elapsed_time
        return elapsed_time

    def __print_output(self, print_task = False):
        if ((self.title is not None) and (not self.__stopwatch_running)):
            print(f'---{self.title}---')

        if (print_task):
            self.__print_simple(-2)
        elif (self.mode == 'simple'):
            self.__print_simple(-1)
        elif (self.mode == 'table'):
            a_prettytable = PrettyTable(['task', 'duration', '% of TOTAL'])
            # only print individual tasks if more than one measurement was taken
            # last task is "TOTAL" and is printed seperatly 
            if (len(self.tasks)>2):
                for i in range(len(self.tasks)-1):
                    self.__add_task_row(a_prettytable, i)
                a_prettytable.add_row([ '--', '--', '--'])

            self.__add_task_row(a_prettytable, -1)
            print(a_prettytable)
            # print(type(self.tasks[0]))
            # print(type(self.durations[0]))
            # print(type(self.timecodes[0]))


    def __padding(self, list, min_length):
        min_padding = len(max(str(list)))
        if (min_padding < min_length):
            min_padding = min_length
        return min_padding

    def __print_simple(self, i):
        elapsed_time = self.durations[i+1]
        min_padding1 = self.__padding(self.tasks, self.min_task_name_length)
        min_padding2 = self.__padding(self.durations, self.min_time_length)
        # print(' ----------------- ')
        s = f'{self.tasks[i]:{min_padding1}} took {(elapsed_time):{min_padding2}.1f} {self.time_unit}'
        print(s)

    def __f_elapsed_time(self, t1, t2):
        return (t2-t1-self.selfexecutiontime)*self.time_units_dict.get(self.time_unit)

    def __add_task_row(self, table, i):
        elapsed_time = self.durations[i]
        task_name = self.tasks[i]
        padding = self.__padding(self.durations[:-1], self.min_time_length)
        percent_of_total = (elapsed_time/self.durations[-1])*100
        # table.add_row([task_name, f'{(elapsed_time):.1f} {self.time_unit}', f'{percent_of_total:{padding}.1f} %'])
        table.add_row([task_name, f'{(elapsed_time):{padding-3}.1f} {self.time_unit}', f'{percent_of_total:5.1f} %'])



def main():
    from time import sleep
    t = stopwatch(title = "test title", mode = 'table', time_unit = 's')
    t.task()
    sleep(0.1)
    t.task(print_task=True)
    sleep(0.2)
    t.task(print_task=True)
    sleep(0.3)
    t.stop('end')

    t = stopwatch(title = '\nTESTING no tasks')
    t.task()
    sleep(0.5)
    t.stop()

    t = stopwatch(title = '\nTESTING no tasks')
    t.t()
    sleep(0.5)
    t.p()

    t = stopwatch(title = '\nTESTING multiple tasks')
    t.task()
    sleep(0.3)
    t.task('0.4')
    sleep(0.4)
    t.task('0.5s')
    sleep(0.5)
    t.stop()
    print(t.task_durations)

    print('####TESTING multiple tasks####')
    tk = []
    tk.append({'title' : '\nmutliple table', 'start' : True})
    tk.append({'title' : '\nmutliple table; task_name specified', 'start' : True, 'task_name' : '0.1s'})
    tk.append({'title' : '\nmutliple table; return_results', 'start' : True, 'return_results' : True})
    tk.append({'title' : '\nmutliple table; time_unit = ''ms''', 'start' : True, 'time_unit' : 'ms'})
    tk.append({'title' : '\nmutliple table; time_unit = ''s''', 'start' : True, 'time_unit' : 's'})
    tk.append({'title' : '\nmutliple table; time_unit = ''min''', 'start' : True, 'time_unit' : 'min'})
    
    tk.append({'title' : '\nmutliple table; print_all_tasks; task_name specified', 
                'start' : True, 'print_all_tasks' : True,
                'task_name' : '0.1s',
                'min_task_name_length' : 10,
                'min_time_length' : 7})
    tk.append({'title' : '\nmutliple table; task_name specified', 
                'start' : True,
                'min_task_name_length' : 10,
                'task_name' : '0.1s',
                'min_time_length' : 7})

    tk.append({'title' : 'mutliple simple', 'start' : True, 'mode' : 'simple', 'print_all_tasks' : True})

    for kwarg in tk:
        t = stopwatch(**kwarg, print_task=True)
        sleep(0.1)
        t.task('0.2s', True)
        sleep(0.2)
        t.task('0.3s')
        sleep(0.3)
        t.task('0.4s')
        sleep(0.4)
        print(t.stop())

    print('####TESTING single tasks####')
    tk = []
    tk.append({'title' : '\nsingle table; time_unit = ''ms''', 'start' : True, 'time_unit' : 'ms', 'mode' : 'simple'})
    tk.append({'title' : '\nsingle table; time_unit = ''ms''', 'start' : True, 'time_unit' : 'ms', 'mode' : 'table'})
    tk.append({'title' : '\nsingle table; time_unit = ''ms''', 'start' : True, 'time_unit' : 'ms'})
    
    for kwarg in tk:
        t = stopwatch(**kwarg)
        sleep(0.1)
        print(t.stop())

    print('\n####TESTING stopwatch on detection#####')
    t = stopwatch()
    t.task()
    # t.stop()


    # print('\n####TESTING stopwatch on detection#####')
    # t = stopwatch(title = 'test')
    # for i in range(1000):
    #     t.task('')
    # t.stop()

    # t.task('0.3')
    # sleep(0.3)
    # t.task('0.3')
    # sleep(0.3)
    # # t.stop()
    # t.task('0.4')
    # sleep(0.4)

    # t.stop()
    #%%




if __name__ == "__main__":
    main()
