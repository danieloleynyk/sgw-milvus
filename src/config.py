import pydantic

from src.milvus import MilvusConfig


class Config(pydantic.BaseSettings):
    milvus_config: MilvusConfig = MilvusConfig()

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
