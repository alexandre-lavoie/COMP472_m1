class Log:
    __log: str

    def __init__(self):
        self.__log = ""

    def add(self, line: str):
        self.__log += str(line) + "\n"

    def label(self, label: str, line: str):
        self.add(label + "\n\n" + str(line) + "\n")

    def __str__(self) -> str:
        return self.__log
