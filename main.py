#Read README.md for more details
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=7000, reload=True)
