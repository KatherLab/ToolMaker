from toolmaker.run import ToolRunResult

from tests.utils import initialize, parametrize_invocation

initialize()


@parametrize_invocation("nsclc", "motor_vehicle_accident", "pediatric_rash")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("nsclc", "motor_vehicle_accident", "pediatric_rash")
def test_response_is_str(invocation: ToolRunResult):
    assert "response" in invocation.result
    response = invocation.result["response"]
    assert isinstance(response, str)
