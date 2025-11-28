from contextlib import asynccontextmanager

from database import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Base
from routers import auth, rag
from sqlalchemy import text


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="APIs", lifespan=lifespan)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    with engine.connect() as conn:
        ok = conn.execute(text("SELECT 1")).scalar() == 1
    return {"status": "ok", "db": ok}


app.include_router(auth.router)
app.include_router(rag.router)
