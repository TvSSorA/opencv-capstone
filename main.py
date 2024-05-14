from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Item(BaseModel):
    username: str
    password: str


@app.get("/", tags=["Root"])
async def root():
    return "Hello World"


@app.get("/user", tags=["user"])
async def user():
    return "User database"


@app.post("/register", tags=["register"])
async def register():
    return "Register"


@app.post("/login", tags=["login"])
async def login(item: Item):
    return item


@app.post("/dashboard", tags=["dashboard"])
async def dashboard():
    return "dashboard"


@app.get("/report", tags=["report"])
async def report():
    return "report"


@app.get("/camera", tags=["camera"])
async def camera():
    return "camera"
