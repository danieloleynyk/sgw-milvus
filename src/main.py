import json
from typing import Annotated, Optional

import pydantic
import pymilvus
import uvicorn

from fastapi import FastAPI, Depends, UploadFile

from src.config import Config
from src.milvus import Milvus


app = FastAPI()


class LoadCollection:
    collection: Optional[pymilvus.Collection] = None

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def __call__(self):
        if not self.collection:
            self.collection = Milvus.get_collection(collection_name=self.collection_name)
            self.collection.load()

        return self.collection


class GetVectorField:
    vector_field: Optional[str] = None

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def __call__(self):
        if not self.vector_field:
            self.vector_field = Milvus.get_vector_field(collection_name=self.collection_name)

        return self.vector_field


@app.on_event('startup')
async def start():
    config = Config()
    Milvus.start_connection(config=config.milvus_config)


@app.post('/load/data/<collection_name>')
async def load_collection_data(collection_name: str, embeddings_file: UploadFile):
    collection = Milvus.get_collection(collection_name=collection_name)

    content = await embeddings_file.read()
    embeddings = [{'word_id': i, **embedding} for i, embedding in enumerate(json.loads(content))]

    collection.insert(embeddings)

    return {'response': 'vectors uploaded successfully'}


class IndexParams(pydantic.BaseModel):
    class Params(pydantic.BaseModel):
        nlist: int = 1024

    metric_type: str = 'L2'
    index_type: str = 'IVF_FLAT'
    params: Params = Params()


@app.post('/index/<collection_name>')
async def index_collection(collection_name: str, index_params: IndexParams = IndexParams()):
    collection = Milvus.get_collection(collection_name=collection_name)

    collection.create_index(
      field_name="vector",
      index_params=index_params.dict()
    )

    return {'response': 'index create successfully.'}

search_collection_param = Annotated[pymilvus.Collection, Depends(LoadCollection(collection_name='words'))]


class SearchParams(pydantic.BaseModel):
    class Params(pydantic.BaseModel):
        nprobe: int = 10

    metric_type: str = 'L2'
    offset: int = 0
    limit: Optional[int] = None
    params: Params = Params()


@app.post('/search')
async def search(
    vector: list[float],
    collection: search_collection_param,
    vector_field: Annotated[str, Depends(GetVectorField(collection_name='words'))],
    search_params: SearchParams = SearchParams(),
    limit: int = 100,
    output_fields: Optional[list[str]] = None,
):
    results = collection.search(
        data=[vector],
        anns_field=vector_field,
        param=search_params.dict(),
        limit=limit,
        output_fields=output_fields,
    )

    return list(results[0])


def main():
    uvicorn.run(app)


if __name__ == '__main__':
    main()
