"""Request / response models. FastAPI uses these to validate input and build /docs."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SensorName = Literal["pm25", "pm10", "co", "co2", "voc", "temp", "humidity"]
ScenarioName = Literal["normal", "smoke_fire", "gas_leak", "poor_air_quality",
                       "spike", "stuck", "noise", "drift", "smoke_stuck"]


class SensorReadings(BaseModel):
    """One reading from each sensor. Ranges reject physically impossible input."""
    pm25: float = Field(..., ge=0, le=1000, description="PM2.5, µg/m³")
    pm10: float = Field(..., ge=0, le=1500, description="PM10, µg/m³")
    co: float = Field(..., ge=0, le=500, description="Carbon monoxide, ppm")
    co2: float = Field(..., ge=0, le=10000, description="Carbon dioxide, ppm")
    voc: float = Field(..., ge=0, le=10000, description="VOC, ppb")
    temp: float = Field(..., ge=-40, le=85, description="Temperature, °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity, %")


class AnalyzeRequest(BaseModel):
    readings: SensorReadings


class SimStartRequest(BaseModel):
    scenario: ScenarioName = "normal"
    sensor: Optional[SensorName] = Field(
        None, description="Which sensor to break (fault scenarios). Random if omitted.")


class AnalysisResponse(BaseModel):
    step: int
    timestamp: str
    readings: dict[str, float]
    decision: str
    alarm_action: str
    event: Optional[dict[str, Any]]
    faulty_sensors: dict[str, Any]
    possible_faults: dict[str, Any]
    corroborating_sensors: list[str]
    trust: dict[str, Any]
    explanation: dict[str, Any]
    truth: Optional[dict[str, str]] = Field(
        None, description="Ground truth - only present in simulation mode")
    restarted: bool = False


class ScenarioInfo(BaseModel):
    id: str
    label: str
    description: str
    needs_sensor: bool