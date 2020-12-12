from abc import ABC, abstractmethod


class AbstractOutput(ABC):

    @abstractmethod
    def write(self, payload: str):
        pass
