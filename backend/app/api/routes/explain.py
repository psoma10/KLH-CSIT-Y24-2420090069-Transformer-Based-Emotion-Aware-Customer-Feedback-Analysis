"""POST {api_v1_prefix}/explain — SHAP token attributions for a review.

Registered on the app router by app/main.py. Explanations are computed
on-demand and are not written to the database (unlike /predict).
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.ratelimit import rate_limit
from app.schemas.emotion import ExplainRequest, ExplainResponse

router = APIRouter()

# The tightest limit in the app: an uncached SHAP run is 6-30 seconds of CPU
# in a threadpool shared with every other endpoint, and the cache is trivially
# bypassed by varying the text.
_explain_limit = rate_limit(limit=10, window_seconds=60, name="/explain")


@router.post("/explain", response_model=ExplainResponse, dependencies=[Depends(_explain_limit)])
async def explain(payload: ExplainRequest, request: Request) -> ExplainResponse:
    shap_service = request.app.state.shap_service
    if shap_service is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded. Train the model first.")
    result, cached = await shap_service.explain(payload.text, payload.label)

    return ExplainResponse(
        explained_label=result["explained_label"],
        label_score=result["label_score"],
        tokens=result["tokens"],
        attributions=result["attributions"],
        base_value=result["base_value"],
        cached=cached,
        truncated=result.get("truncated", False),
    )
