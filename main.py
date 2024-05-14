from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    # name: str
    # price: float
    # ready: int
    username: str
    password: str


@app.get("/")
async def home():
    return "Hello World"


@app.post("/login")
async def login(item: Item):
    return item
