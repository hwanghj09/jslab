from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
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
