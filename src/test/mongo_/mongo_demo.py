from pymongo import MongoClient
from typing import Dict, List, Optional


class MongoDemo:
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017/",
        db_name: str = "demo_db",
        collection_name: str = "users",
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # ---------- 插入 ----------

    def insert_one(self, data: Dict) -> str:
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def insert_many(self, data_list: List[Dict]) -> List[str]:
        result = self.collection.insert_many(data_list)
        return [str(_id) for _id in result.inserted_ids]

    # ---------- 查询 ----------

    def find_one(self, query: Dict, projection: Optional[Dict] = None) -> Optional[Dict]:
        return self.collection.find_one(query, projection)

    def find_many(
        self,
        query: Dict,
        projection: Optional[Dict] = None,
        sort: Optional[List] = None,
        skip: int = 0,
        limit: int = 0,
    ) -> List[Dict]:
        cursor = self.collection.find(query, projection)

        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    # ---------- 是否存在 ----------

    def exists(self, query: Dict) -> bool:
        return self.collection.count_documents(query, limit=1) > 0

    # ---------- 更新 ----------

    def update_one(self, query: Dict, update_data: Dict) -> int:
        result = self.collection.update_one(
            query,
            {"$set": update_data},
        )
        return result.modified_count

    def update_many(self, query: Dict, update_data: Dict) -> int:
        result = self.collection.update_many(
            query,
            {"$set": update_data},
        )
        return result.modified_count

    # ---------- 删除 ----------

    def delete_one(self, query: Dict) -> int:
        result = self.collection.delete_one(query)
        return result.deleted_count

    def delete_many(self, query: Dict) -> int:
        result = self.collection.delete_many(query)
        return result.deleted_count

    # ---------- 统计 ----------

    def count(self, query: Dict = None) -> int:
        return self.collection.count_documents(query or {})

    # ---------- 清空集合 ----------

    def clear(self) -> None:
        self.collection.delete_many({})

    # ---------- 关闭连接 ----------

    def close(self):
        self.client.close()


# ================== Demo 演示 ==================

if __name__ == "__main__":
    mongo = MongoDemo()

    # 清空旧数据
    mongo.clear()

    print("=== 插入数据 ===")
    mongo.insert_one({"name": "Alice", "age": 25})
    mongo.insert_many(
        [
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 28},
        ]
    )

    print("\n=== 查询一条 ===")
    print(mongo.find_one({"name": "Alice"}))

    print("\n=== 查询多条（age >= 28）===")
    print(mongo.find_many({"age": {"$gte": 28}}))

    print("\n=== 是否存在 ===")
    print("Alice exists:", mongo.exists({"name": "Alice"}))
    print("Tom exists:", mongo.exists({"name": "Tom"}))

    print("\n=== 更新 ===")
    mongo.update_one({"name": "Alice"}, {"age": 26})
    mongo.update_many({"age": {"$gte": 28}}, {"vip": True})
    print(mongo.find_many({}))

    print("\n=== 统计 ===")
    print("Total count:", mongo.count())
    print("VIP count:", mongo.count({"vip": True}))

    print("\n=== 排序 + 分页 ===")
    users = mongo.find_many(
        {},
        sort=[("age", -1)],
        skip=0,
        limit=2,
    )
    print(users)

    print("\n=== 删除 ===")
    mongo.delete_one({"name": "Bob"})
    print(mongo.find_many({}))

    mongo.close()
