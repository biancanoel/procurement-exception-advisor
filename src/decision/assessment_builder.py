"""Create initial emergency procurement assessments."""

from decision.emergency_criteria import get_emergency_criteria
from models.assessment import (
    CriterionResult,
    EmergencyProcurementAssessment,
    EmergencyVerification,
)
from models.cases import EmergencyCaseInput
from models.criteria import CriterionStatus

def create_initial_assessment(
    case: EmergencyCaseInput,
) -> EmergencyProcurementAssessment:
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
  verification = EmergencyVerification(
      case_id=case.case_id,
      emergency_is_verified=None,
      criterion_results=criterion_results,
      rationale="The emergency request has not yet been evaluated.",
      confidence=0.0,
  )
  return EmergencyProcurementAssessment(
      case_id=case.case_id,
      emergency_verification=verification,
      audit_readiness=None,
  )
