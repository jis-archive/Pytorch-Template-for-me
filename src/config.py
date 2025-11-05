import json
import logging
from collections import defaultdict

class ConfigParser:
    """
    ConfigParser 클래스
    - JSON 파일 기반 설정 관리
    - CLI 옵션 덮어쓰기 가능
    - 로깅 지원
    """
    def __init__(self, config_dict):
        self.config = config_dict
        self.args = None

    @classmethod
    def from_json(cls, filepath):
        """JSON 파일을 읽어 ConfigParser 객체 생성"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_args(cls, parser, options=[]):
        """
        CLI 인자를 읽어 JSON config에 덮어쓰기
        options: list of namedtuple('CustomArgs', 'flags type target')
        """
        args = parser.parse_args()
        config = cls.from_json(args.config)
        config.args = args

        # CLI 값으로 config 덮어쓰기
        for opt in options:
            for flag in opt.flags:
                arg_name = flag.lstrip('-').replace('-', '_')
                if hasattr(args, arg_name):
                    value = getattr(args, arg_name)
                    if value is not None:
                        config.set_value(opt.target, value)

        return config

    def set_value(self, path, value):
        """
        점(.) 또는 세미콜론(;) 구분 경로를 통해 config 값 변경
        ex) 'optimizer;args;lr'
        """
        keys = path.replace('.', ';').split(';')
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def get(self, path, default=None):
        """점(.) 또는 세미콜론(;) 구분 경로로 값 가져오기"""
        keys = path.replace('.', ';').split(';')
        d = self.config
        try:
            for k in keys:
                d = d[k]
            return d
        except KeyError:
            return default

    def get_logger(self, name, level=logging.INFO):
        """간단한 logger 생성"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        return logger

    def __getitem__(self, key):
        """config[key] 형태 접근 지원"""
        return self.config[key]

    def __repr__(self):
        return json.dumps(self.config, indent=2, ensure_ascii=False)
