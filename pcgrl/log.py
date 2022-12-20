import csv
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class

    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """    

    def __init__(
        self,
        filename: str = "",
        path : str = "",
        fieldsnames = (),
        header: Optional[Dict[str, Union[float, str]]] = None#,
        #extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        self.filename = ""
        if os.path.isdir(path):
            filename = os.path.join(path, filename)
            self.filename = filename
                    
        #if not filename.endswith("results.csv"):
        #    if os.path.isdir(path):
        #        filename = os.path.join(path, "results.csv")
        #    else:
        #        filename = filename + "." + "results.csv"
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, "wt", newline="\n")
        #self.file_handler.write("#%s\n" % json.dumps(header))
        #self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + extra_keys)
        self.logger = csv.DictWriter(self.file_handler, fieldnames=fieldsnames) #+ extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()