"""Create initial emergency procurement assessments."""

from decision.emergency_criteria import get_emergency_criteria
from models.assessment import (
    CriterionResult,
    EmergencyAssessment,
    FinalRecommendation,
)
from models.cases import EmergencyCaseInput
from models.criteria import CriterionStatus

def create_initial_assessment(
    case: EmergencyCaseInput,
) -> EmergencyAssessment:
  """Create a blank structured assessment for one case."""
  criterion_results = [
      CriterionResult(
          criterion_id=criterion.criterion_id,
          status=CriterionStatus.NOT_EVALUATED,
          rationale="This criterion has not yet been evaluated.",
          confidence=0.0,
      )
      for criterion in get_emergency_criteria()
  ]
  return EmergencyAssessment(
     case_id = case.case_id,
     recommendation = FinalRecommendation.HUMAN_REVIEW_REQUIRED,
     executive_summary= (
        "Assessment has been initialized but not yet completed."
      ),
      classification = "Not yet classified",
      criterion_results = criterion_results,
      overall_confidence = 0.0,
      requires_human_review = True,
      human_review_reason = "The emergency request has not yet been evaluated."
  )