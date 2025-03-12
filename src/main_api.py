import uvicorn
from src.config.settings import API_SETTINGS

if __name__ == "__main__":
    uvicorn.run(
        "src.api.service:app",
        host=API_SETTINGS["HOST"],
        port=API_SETTINGS["PORT"],
        workers=API_SETTINGS["WORKERS"],
        reload=API_SETTINGS["DEBUG"]
    )
