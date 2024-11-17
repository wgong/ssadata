from .base import VannaBase

SQL_DIALECTS = [
    "Snowflake",
    "SQLite",
    "PostgreSQL",
    "MySQL",
    "ClickHouse",
    "Oracle",
    "BigQuery",
    "DuckDB",
    "Microsoft SQL Server",
    "Presto",
    "Hive",
]

VECTOR_DB_LIST = [
    "chromadb", 
    "marqo", 
    "opensearch", 
    "pinecone", 
    "qdrant",
    "faiss",
    "milvus",
    "pgvector",
    "weaviate",
]


__all__ = ['SQL_DIALECTS', 'VECTOR_DB_LIST', 'VannaBase']