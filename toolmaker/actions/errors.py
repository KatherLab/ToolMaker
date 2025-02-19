from .actions import Observation


class FunctionCallError(Exception):
    """Error when calling a function"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class FunctionCallErrorObservation(Observation):
    content: str = "An error occurred while calling the function."
    error: str
