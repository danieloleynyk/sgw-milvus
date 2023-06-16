import pymilvus

from . import Schema


class WordSchema(Schema):
    collection_name: str = 'words'
    vector_field: str = 'vector'

    @staticmethod
    def get_schema() -> pymilvus.CollectionSchema:
        word_id = pymilvus.FieldSchema(
            name="word_id",
            dtype=pymilvus.DataType.INT64,
            is_primary=True,
        )

        word = pymilvus.FieldSchema(
            name="word",
            dtype=pymilvus.DataType.VARCHAR,
            max_length=200,
        )

        word_vector = pymilvus.FieldSchema(
            name=WordSchema.vector_field,
            dtype=pymilvus.DataType.FLOAT_VECTOR,
            dim=300,
        )

        schema = pymilvus.CollectionSchema(
            fields=[word_id, word, word_vector],
            enable_dynamic_field=True,
        )

        return schema
