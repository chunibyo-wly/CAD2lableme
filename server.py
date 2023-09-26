from datetime import timedelta, datetime
from typing import List
from typing_extensions import Annotated

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, FileResponse
from jose import JWTError, jwt

from cad2labelme import proceed

SECRET_KEY = "XS25umz&o1@R\_B4'ZBy{19/<{tkUaAi6cPYLgS[33u?"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
USERS = set()


# 创建访问令牌
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    hashed_password = form_data.password
    if not hashed_password == "hzTJNr0iNAFfx6J95oyA":
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    USERS.add(form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in USERS:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


@app.post("/cad/")
async def cad_reconstruction(
    height: int = Form(),
    file: UploadFile = File(),
    username: str = Depends(get_current_user),
):
    temp_file = "./tmp/tmp.dwg"
    with open(temp_file, "wb") as f:
        f.write(await file.read())
    result_file = proceed(temp_file, height * 1000)
    print(result_file)
    return FileResponse(result_file)
    # return {"filenames": file.filename, "height": height}
