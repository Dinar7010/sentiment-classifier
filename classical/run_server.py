import uvicorn
#http://127.0.0.1:8000/docs
if __name__=='__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)