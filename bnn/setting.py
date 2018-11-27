"""データの保存先ディレクトリなど横断的に使用する定数値を設定する python file
"""

import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
