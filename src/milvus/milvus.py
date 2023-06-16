import pymilvus
from pymilvus import connections, MilvusException

from . import MilvusConfig
from .schemas import Schema, WordSchema
from .exceptions import SchemaDoesNotExist


class Milvus:
    collections: dict[str, Schema] = {
        WordSchema.collection_name: WordSchema
    }

    @staticmethod
    def start_connection(config: MilvusConfig):
        connections.connect(
            alias=config.alias,
            user=config.username,
            password=config.password,
            host=config.hostname,
            port=config.port,
        )

        if config.drop_collections:
            Milvus.__drop_collections()

    @staticmethod
    def __drop_collections():
        for collection in Milvus.collections:
            try:
                pymilvus.utility.drop_collection(collection_name=collection)
            except MilvusException as e:
                print(f'failed dropping collection: {collection} due to: {e}')

    @staticmethod
    def get_collection(collection_name: str, *args, **kwargs) -> pymilvus.Collection:
        if schema := Milvus.collections.get(collection_name):
            return pymilvus.Collection(name=collection_name, schema=schema.get_schema(), *args, **kwargs)

        raise SchemaDoesNotExist(f'a schema for {collection_name} does not exist.')

    @staticmethod
    def get_vector_field(collection_name: str):
        if schema := Milvus.collections.get(collection_name):
            return schema.vector_field

        raise SchemaDoesNotExist(f'a schema for {collection_name} does not exist.')
