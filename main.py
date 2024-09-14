from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, WebSocket
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Union, List
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import numpy as np
from matplotlib import cm
import base64
import io
import matplotlib.pyplot as plt
import asyncio

# 기본 설정
Base = declarative_base()
app = FastAPI()
templates = Jinja2Templates(directory="public")

# 데이터베이스 설정
SQLALCHEMY_DATABASE_URL = "mysql://root:qwaszx77^^@svc.sel4.cloudtype.app:31994/sc"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# 비밀번호 해시 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 설정
SECRET_KEY = "asdlkjfj01u39rlkaskcmklsajfea79803423597802349083725235697mlk103u0590aksdlflksndkdkckkcfkjdfkjderkjdr"  # 강력한 비밀 키로 변경하세요
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30
app.mount("/image", StaticFiles(directory="image"), name="image")
# 데이터베이스 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# JWT 생성 및 검증 함수
def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("email")
        if email is None:
            raise JWTError
        return {"email": email}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# Comment 모델 정의
class CommentModel(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, nullable=True)  # 포스트가 없는 경우를 대비하여 nullable=True
    user_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    user = relationship("UserModel", back_populates="comments")

# User 모델에 comments 관계 추가
class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    
    comments = relationship("CommentModel", back_populates="user")

# Pydantic 스키마 정의
class CommentCreateSchema(BaseModel):
    post_id: Optional[int] = None
    user_id: int
    content: str

class CommentResponseSchema(BaseModel):
    id: int
    post_id: Optional[int]
    user_id: int
    content: str
    created_at: datetime

    class Config:
        orm_mode = True

# 사용자 스키마
class UserCreateSchema(BaseModel):
    name: str
    email: EmailStr 
    password: str
    confirm_password: str

    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# 라우트 정의
@app.get("/")
@app.get("/index")
async def main(request: Request):
    token = request.cookies.get("access_token")
    islogin = False
    token_data = {"email": None}

    if token:
        islogin = True
        try:
            token_data = verify_token(token)
        except HTTPException:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return templates.TemplateResponse("index.html", {"request": request, "email": token_data["email"], "islogin": islogin})

@app.get("/실험/{name}")
async def experiment(request: Request, name: str, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    islogin = False
    id=0
    token_data = {"email": None}
    if name == "불꽃실험":
        id=1
    elif name=="전기회로-직":
        id=2
    elif name=="전기회로-병":
        id=3
    elif name=="태양계":
        id=4
    elif name=="식물과에너지":
        id=5
    elif name=="용해도":
        id=6
    elif name=="전기회로":
        id=7    
    if token:
        islogin = True
        try:
            token_data = verify_token(token)
        except HTTPException:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # 댓글 조회
    comments = db.query(CommentModel).filter(CommentModel.post_id == id).all()
    
    # 댓글을 JSON 형식으로 변환 (optional)
    comment_data = [{"content": comment.content, "created_at": comment.created_at} for comment in comments]

    return templates.TemplateResponse(
        f"{name}.html", 
        {
            "request": request,
            "email": token_data["email"],
            "islogin": islogin,
            "comments": comment_data  # 댓글 데이터를 템플릿에 전달
        }
    )
@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup")
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def register_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirmPassword: str = Form(...),
    db: Session = Depends(get_db)
):
    user_data = UserCreateSchema(
        name=name,
        email=email,
        password=password,
        confirm_password=confirmPassword
    )
    try:
        user_data.validate_passwords()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    existing_user_email = db.query(UserModel).filter(UserModel.email == user_data.email).first()
    if existing_user_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    db_user = UserModel(
        name=user_data.name,
        email=user_data.email,
        password=hashed_password
    )

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database integrity error")

    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    access_token = create_access_token(data={"email": user_data.email}, expires_delta=access_token_expires)
    
    response = RedirectResponse(url="/index", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_DAYS*86400)
    
    return response

@app.post("/login")
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(UserModel).filter(UserModel.email == email).first()

    if not user or not pwd_context.verify(password, user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password")
    
    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    access_token = create_access_token(data={"email": user.email}, expires_delta=access_token_expires)
    
    response = RedirectResponse(url="/index", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_DAYS*86400)
    
    return response

@app.post("/comments/{post_id}")
async def create_comment(
    request: Request,
    post_id: int,
    content: str = Form(...),
    db: Session = Depends(get_db)
):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    token_data = verify_token(token)
    user_email = token_data["email"]
    user = db.query(UserModel).filter(UserModel.email == user_email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db_comment = CommentModel(post_id=post_id, user_id=user.id, content=content)
    db.add(db_comment)
    try:
        db.commit()
        db.refresh(db_comment)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database integrity error")
    
    return db_comment
@app.get("/logout")
async def logout_user(request: Request):
    response = RedirectResponse(url="/index", status_code=303)
    response.delete_cookie(key="access_token")
    return response



Re = 200
nx, ny = 105, 45
ly = ny - 1
uLB = 0.01
cx, cy, r = nx // 6, ny // 2, ny // 11
nulb = 1 * uLB * r / Re
omega = 1 / (3 * nulb + 0.5)

# Lattice constants
ci = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])
wi = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])

col_left = np.array([0, 4, 7])
col_rest = np.array([1, 3, 8])
col_right = np.array([2, 5, 6])
col_wall = np.array([2, 3, 0, 1, 5, 6, 7, 6, 8])

def macroscopic(fin):
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += ci[i, 0] * fin[i, :, :]
        u[1, :, :] += ci[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u

def equilibrium(rho, u):
    usqr = (3/2) * (u[0]**2 + u[1]**2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        cu = 3 * (ci[i, 0] * u[0, :, :] + ci[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * wi[i] * (1 + cu + 0.5 * cu**2 - usqr)
    return feq

def plot_to_base64(u):
    plt.figure(figsize=(10, 5))
    speed = np.sqrt(u[0]**2 + u[1]**2)
    plt.imshow(speed.transpose(), cmap=cm.Reds)
    plt.colorbar()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize
    obstacle = np.zeros((nx, ny), bool)
    for i in range(nx):
        for j in range(ny):
            if (i - cx)**2 + (j - cy)**2 <= r**2:
                obstacle[i, j] = True

    u = np.zeros((2, nx, ny))
    u[0, :, :] = uLB
    fin = equilibrium(1, u)

    # Main loop
    time = 0
    while True:
        fin[col_right, -1, :] = fin[col_right, -2, :]
        rho, u = macroscopic(fin)
        u[0, 0, :] = uLB
        u[1, 0, :] = 0
        rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(fin[col_rest, 0, :], axis=0) + 2 * np.sum(fin[col_right, 0, :], axis=0))
        feq = equilibrium(rho, u)
        fin[col_left, 0, :] = feq[col_left, 0, :] + fin[col_right, 0, :] - feq[col_right, 0, :]
        fout = fin - omega * (fin - feq)
        for i in range(9):
            fout[i, obstacle] = fin[col_wall[i], obstacle]
        for i in range(9):
            fin[i, :, :] = np.roll(np.roll(fout[i, :, :], ci[i, 0], axis=0), ci[i, 1], axis=1)
        
        if time % 10 == 0:  # Send update every 10 iterations
            plot_data = plot_to_base64(u)
            await websocket.send_text(plot_data)
        
        time += 1
        await asyncio.sleep(0.1)  # Small 