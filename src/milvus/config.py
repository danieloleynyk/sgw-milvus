import pydantic


class Config(pydantic.BaseSettings):
    username: str = pydantic.Field(..., env='MILVUS_USERNAME')
    password: str = pydantic.Field(..., env='MILVUS_PASSWORD')
    hostname: str = pydantic.Field(..., env='MILVUS_HOSTNAME')
    port: int = pydantic.Field(..., env='MILVUS_PORT')
    alias: str = 'default'

    drop_collections: bool = pydantic.Field(..., env='MILVUS_DROP_COLLECTIONS')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
