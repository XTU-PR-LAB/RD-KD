import sys
import os
import time
from datetime import datetime

class Logger:
    def __init__(self, log_dir, log_file_name=None):
        if log_file_name is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_name = f'training_log_{current_time}.txt'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file_path = os.path.join(log_dir, log_file_name)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.enabled = False
    
    def enable(self):
        if not self.enabled:
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            sys.stdout = self
            sys.stderr = self
            self.enabled = True
            print(f"日志已启用，日志文件保存在：{self.log_file_path}")
    
    def disable(self):
        if self.enabled:
            self.enabled = False
            if sys.stdout is self:
                sys.stdout = self.original_stdout
            if sys.stderr is self:
                sys.stderr = self.original_stderr
            self.log_file.close()
            print(f"日志已禁用，日志文件保存在：{self.log_file_path}")
    
    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.original_stdout.flush()
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.flush()
    
