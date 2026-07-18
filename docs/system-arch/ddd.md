---
icon: simple/markdown
---

# FastAPI + SQLModel DDD分层

整体架构
```
src/
├── domain/
│   ├── entities/stock.py       # 充血实体
│   └── repos/stock_repo.py     # repo 抽象接口
├── application/usecase/stock_deduct.py  # UseCase
├── infra/
│   ├── orm_models/stock_orm.py # ORM表模型
│   └── repo_impl/stock_repo_impl.py # repo实现
└── main.py # FastAPI入口
```

## 1. domain/entities/stock.py 充血实体
无ORM、无DB，只处理自身业务逻辑
```python
class Stock:
    def __init__(self, stock_id: int, quantity: int):
        self.stock_id = stock_id
        self.quantity = quantity

    # 单实体业务行为
    def deduct(self, num: int):
        if self.quantity < num:
            raise ValueError("库存不足")
        self.quantity -= num
```

## 2. domain/repos/stock_repo.py 仓储抽象（repos）
只定义方法，不碰数据库
```python
from abc import ABC, abstractmethod
from domain.entities.stock import Stock

class StockRepo(ABC):
    # 悲观锁查询
    @abstractmethod
    async def get_by_id_lock(self, stock_id: int) -> Stock:
        ...

    # 保存实体
    @abstractmethod
    async def save(self, stock: Stock) -> None:
        ...
```

## 3. infra/orm_models/stock_orm.py ORM模型
仅映射数据库表
```python
from sqlmodel import SQLModel, Field

class StockORM(SQLModel, table=True):
    stock_id: int = Field(primary_key=True)
    quantity: int
```

## 4. infra/repo_impl/stock_repo_impl.py repo_impl 实现
实现上面抽象，封装ORM、实体转换、SQL锁
```python
from sqlmodel import Session, select
from domain.repos.stock_repo import StockRepo
from domain.entities.stock import Stock
from infra.orm_models.stock_orm import StockORM

class StockRepoImpl(StockRepo):
    def __init__(self, session: Session):
        self.db = session

    async def get_by_id_lock(self, stock_id: int) -> Stock:
        # ORM悲观锁查询
        sql = select(StockORM).where(StockORM.stock_id == stock_id).with_for_update()
        orm_obj = self.db.exec(sql).one()
        # ORM -> Domain Entity
        return Stock(stock_id=orm_obj.stock_id, quantity=orm_obj.quantity)

    async def save(self, stock: Stock):
        # Domain Entity -> ORM
        db_row = StockORM(stock_id=stock.stock_id, quantity=stock.quantity)
        self.db.merge(db_row)
        self.db.commit()
```

## 5. application/usecase/stock_deduct.py UseCase
处理锁、事务、调度repo与实体
```python
from domain.repos.stock_repo import StockRepo

class DeductStockUseCase:
    def __init__(self, stock_repo: StockRepo):
        self.repo = stock_repo

    async def execute(self, stock_id: int, buy_num: int):
        # 1. 事务内带锁查询领域实体
        stock = await self.repo.get_by_id_lock(stock_id)
        # 2. 调用充血实体执行业务逻辑
        stock.deduct(buy_num)
        # 3. repo持久化
        await self.repo.save(stock)
        return stock
```

## 6. main.py FastAPI 调用演示
```python
from fastapi import FastAPI
from sqlmodel import create_engine, Session
from application.usecase.stock_deduct import DeductStockUseCase
from infra.repo_impl.stock_repo_impl import StockRepoImpl

app = FastAPI()
engine = create_engine("sqlite:///test.db")

@app.post("/stock/deduct")
async def deduct_api(stock_id: int, num: int):
    with Session(engine) as session:
        # 实例化 repo_impl 实现类，注入抽象接口
        repo = StockRepoImpl(session)
        use_case = DeductStockUseCase(repo)
        result = await use_case.execute(stock_id, num)
    return {"stock_id": result.stock_id, "left_quantity": result.quantity}
```

# 区分 repos / repo_impl 关键点对照
1. `domain/repos/StockRepo`
    - 抽象协议，只规定“能查锁库存、能保存”
    - 上层UseCase只依赖这个抽象，看不到SQLModel
2. `infra/repo_impl/StockRepoImpl`
    - 实现抽象接口，真正写ORM语句
    - 负责数据库查询、ORM与领域实体互转
    - 唯一允许操作 `StockORM` 的地方

# 调用链路
路由 → UseCase(依赖抽象StockRepo) → 运行时注入StockRepoImpl → 操作StockORM读写库