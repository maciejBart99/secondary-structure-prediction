from core.abstract_output import AbstractOutput


class BasicOutput(AbstractOutput):

    def __init__(self, log_file=None, log_stdout=True):
        self.log_file = log_file
        self.log_stdout = log_stdout

    def write(self, payload: str):
        if self.log_stdout:
            print(payload)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(payload + '\n')

