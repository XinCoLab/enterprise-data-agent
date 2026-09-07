import logging

logging.basicConfig(level=logging.WARNING)  # 显示 INFO 及以上级别的日志

logging.info("收到用户问题")
logging.warning("模型响应时间较长")
logging.error("模型请求超时")
