from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def demo():
  return {'example':'This is an example code', 'data':1}

