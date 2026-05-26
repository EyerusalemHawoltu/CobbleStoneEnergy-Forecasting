from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import pipeline

router = APIRouter()


@router.get("/report")
def qa_report():
    """Data quality report: missingness, duplicates, outliers, LLM-proposed rules."""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    return pipeline.get_qa_summary()
