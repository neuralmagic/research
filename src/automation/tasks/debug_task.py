from automation.tasks import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
import os

class DebugTask(BaseTask):

    def __init__(
        self,
        time_in_sec: int,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        *args,
        **kwargs,
    ):
        super().__init__(docker_image=docker_image, *args, **kwargs)
        self.time_in_sec = time_in_sec
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "debug_script.py")


    def script(self):
        from automation.tasks.scripts.debug_script import main
        main()


    def get_arguments(self):
        return {
            "Args": {
                "time_in_sec": self.time_in_sec,
            },
        }
