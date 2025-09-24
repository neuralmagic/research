from clearml import Task
import time

try:
    from clearml import Task
    clearml_available = True
except ImportError:
    clearml_available = False


def countdown(time_in_sec):
    while time_in_sec:
        mins, secs = divmod(time_in_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        time_in_sec -= 1

    print("stop")


def main(configurations=None, args=None):
    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)["Args"]

    time_in_sec = int(args["time_in_sec"])

    print("Debugging task initiated", flush=True)
    countdown(time_in_sec)


if __name__ == '__main__':
    main()
