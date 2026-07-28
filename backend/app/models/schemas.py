from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"
    id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email    = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    documents  = relationship("Document", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"
    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id   = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    filename   = Column(String, nullable=False)
    subject    = Column(String, nullable=False)
    status     = Column(String, default="processing")  # processing | ready | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    owner      = relationship("User", back_populates="documents")

class QuizResult(Base):
    __tablename__ = "quiz_results"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    subject     = Column(String, nullable=False)
    question    = Column(Text, nullable=False)
    user_answer = Column(Text, nullable=False)
    score       = Column(String, nullable=False)
    feedback    = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)