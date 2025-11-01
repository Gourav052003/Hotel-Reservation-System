import traceback
import sys

class CustomException(Exception):

    def __init__(self,error_message,error_detail:sys):

        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_detail)

    @staticmethod
    def get_detailed_error_message(error_message,error_detail:sys):

        _,_, execution_traceback = traceback.sys.exc_info()
        file_name = execution_traceback.tb_frame.f_code.co_filename # filename in which error occured
        line_no = execution_traceback.tb_lineno # Line in which error occured

        return f"Error occured in {file_name}, line {line_no}: {error_message}"
    
    def __str__(self):
        # gives text representation of error message
        return self.error_message
