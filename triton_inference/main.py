from fastapi import FastAPI
from routers import router_var

app = FastAPI(title="BGE-M3 Triton Embeddings Gateway")
app.include_router(router_var)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)