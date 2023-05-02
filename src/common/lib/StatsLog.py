import os
import logging

class StatsLogger(object):
    """
    class that enable to apture  logs into mutiple files.
    """
    formatter = None
    handler = None
    specified_logger = None

    def SetFileName(self, file_name, log_name ="StatsType", level=logging.INFO):
        self.formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
        self.handler = logging.FileHandler(file_name)
        self.handler.setFormatter(self.formatter)
        self.specified_logger = logging.getLogger(log_name)
        self.specified_logger.setLevel(level)
        self.specified_logger.addHandler(self.handler)
    def line(self,line):
        self.specified_logger.info(line)
    def vector(self, Name, Vector):
        num_list = [ f for f in Vector]
        message = Name + ' : ' + str(num_list)
        self.specified_logger.info(message)


Stats_log = StatsLogger()    # ('StatsLog', 'StatsLog.txt')