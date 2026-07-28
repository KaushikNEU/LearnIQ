import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.rag.ingestion import parse_document, chunk_text
from app.rag.vectorstore import store_chunks, list_subjects, delete_subject

router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form(...)
):
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save file temporarily
    temp_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Parse → chunk → embed → store
        text = parse_document(temp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        chunks = chunk_text(text, subject, file.filename)
        stored = await store_chunks(chunks, subject)

        return {
            "status": "success",
            "filename": file.filename,
            "subject": subject,
            "chunks_stored": stored
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/subjects")
def get_subjects():
    return {"subjects": list_subjects()}

@router.delete("/{subject}")
def remove_subject(subject: str):
    delete_subject(subject)
    return {"status": "deleted", "subject": subject}