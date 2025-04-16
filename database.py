import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
import os

Base = declarative_base()
DATABASE_URL = "sqlite:///speaker_recognition.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Speaker(Base):
    __tablename__ = 'speakers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    voice_sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_updated = Column(DateTime, onupdate=datetime.datetime.utcnow)

class AudioRecording(Base):
    __tablename__ = 'audio_recordings'
    
    id = Column(Integer, primary_key=True)
    speaker_id = Column(Integer, nullable=False)
    file_path = Column(String(255), nullable=False)
    duration = Column(Float)
    sample_rate = Column(Integer)
    mfcc_features = Column(LargeBinary)  # Serialized numpy array
    recording_date = Column(DateTime, default=datetime.datetime.utcnow)

class RecognitionResult(Base):
    __tablename__ = 'recognition_results'
    
    id = Column(Integer, primary_key=True)
    audio_file = Column(String(255), nullable=False)
    predicted_speaker_id = Column(Integer)
    actual_speaker_id = Column(Integer)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    noise_level = Column(Float, default=0.0)

def init_db(db_url='sqlite:///speaker_recognition.db'):
    if db_url.startswith('sqlite'):
        engine = create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        engine = create_engine(db_url)
    
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def get_db_session() -> Session:
    return SessionLocal()