from abc import ABC, abstractmethod

import pymilvus


class Schema(ABC):
    collection_name: str
    vector_field: str

    @staticmethod
    @abstractmethod
    def get_schema() -> pymilvus.CollectionSchema:
        raise NotImplementedError
