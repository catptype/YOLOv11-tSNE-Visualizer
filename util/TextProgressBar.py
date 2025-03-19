import time

class TextProgressBar():
    """
    A simple text-based progress bar for tracking the progress of a task.

    Attributes:
        total_step (int): The total number of steps in the task.
        length (int): The length of the progress bar.
        time_delay (float): The time delay between updates.
    
    Methods:
        add_step(num): Add a specified number of steps to the progress bar.

    Example:
        To create a progress bar for a task with 100 steps, you can use the following code:

        ```python
        progress = TextProgressBar(total_step=100)
        for step in range(100):
            # Perform a step of the task
            # ...
            progress.add_step(1)
        ```

    Note:
        This progress bar does not handle multi-threading or multiprocessing scenarios.
    """
    def __init__(self, total_step:int, length:int=40, time_delay:float=0.1):
        """
        Args:
            total_step (int): The total number of steps in the task.
            length (int): The length of the progress bar in characters (default is 40).
            time_delay (float): The time delay (in seconds) between updates (default is 0.1).
        """
        if not isinstance(total_step, int):
            raise ValueError("total_step must be an integer")
        self.total_step = total_step
        self.length = length
        self.time_delay = time_delay
        self.__cur_step = 0
        self.__start_time = time.time()
        self.__prev_time = self.__start_time
        
        self.__print_progress_bar()

    # Private method
    def __eta_calculation(self):
        # Initialize ETA
        elapsed_time = time.time() - self.__start_time
        if self.__cur_step > 0:
            initial_eta = (elapsed_time / self.__cur_step) * (self.total_step - self.__cur_step)
        else:
            initial_eta = 0

        # Post processing ETA for more accurate ETA
        step_time = time.time() - self.__prev_time
        if self.__cur_step < self.total_step:
            eta = int(initial_eta - step_time)
        else:
            eta = 0  # When all steps are completed, ETA is 0

        self.__prev_time = time.time()
        return max(eta, 0)  # Ensure that ETA is not negative
    
    def __print_progress_bar(self):
        eta = self.__eta_calculation()
        progress = (self.__cur_step / self.total_step)
        bars = '█' * int(self.length * progress)
        spaces = ' ' * (self.length - len(bars))
        print(f'\r{self.__cur_step}/{self.total_step} [{bars}{spaces}] {int(progress * 100)}% | ETA: {eta}s\t', end='', flush=True)
        time.sleep(self.time_delay)
    
    # Public method
    def add_step(self, num=1):
        """
        Add a specified number of steps to the progress bar.

        Args:
            num (int): The number of steps to add.

        Example:
            To add 5 steps to the progress bar, you can use the following code:

            ```python
            progress.add_step(5)
            ```
        """
        self.__cur_step += num
        self.__print_progress_bar()