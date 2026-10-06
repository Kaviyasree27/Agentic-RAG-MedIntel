from typing import Dict, Any, List

class LabInterpreter:
    """Interprets clinical laboratory tests against standard adult reference ranges."""

    REFERENCE_RANGES = {
        "potassium": {"min": 3.5, "max": 5.0, "unit": "mEq/L", "critical_low": 2.8, "critical_high": 6.2, "name": "Serum Potassium"},
        "sodium": {"min": 135.0, "max": 145.0, "unit": "mEq/L", "critical_low": 120.0, "critical_high": 160.0, "name": "Serum Sodium"},
        "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "critical_low": 0.2, "critical_high": 4.0, "name": "Serum Creatinine"},
        "bun": {"min": 7.0, "max": 20.0, "unit": "mg/dL", "critical_low": 3.0, "critical_high": 80.0, "name": "Blood Urea Nitrogen (BUN)"},
        "hba1c": {"min": 4.0, "max": 5.6, "unit": "%", "critical_low": 3.0, "critical_high": 14.0, "name": "Hemoglobin A1c"},
        "fasting_glucose": {"min": 70.0, "max": 99.0, "unit": "mg/dL", "critical_low": 50.0, "critical_high": 400.0, "name": "Fasting Blood Glucose"},
        "inr": {"min": 0.8, "max": 1.2, "unit": "ratio", "critical_low": 0.5, "critical_high": 4.5, "name": "International Normalized Ratio (INR)"},
        "alt": {"min": 7.0, "max": 56.0, "unit": "U/L", "critical_low": 0.0, "critical_high": 500.0, "name": "Alanine Aminotransferase (ALT)"},
        "ast": {"min": 10.0, "max": 40.0, "unit": "U/L", "critical_low": 0.0, "critical_high": 500.0, "name": "Aspartate Aminotransferase (AST)"},
        "wbc": {"min": 4.5, "max": 11.0, "unit": "10^3/uL", "critical_low": 2.0, "critical_high": 30.0, "name": "White Blood Cell Count (WBC)"},
        "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL", "critical_low": 7.0, "critical_high": 20.0, "name": "Hemoglobin"},
        "platelets": {"min": 150.0, "max": 450.0, "unit": "10^3/uL", "critical_low": 50.0, "critical_high": 1000.0, "name": "Platelets"}
    }

    @classmethod
    def interpret(cls, test_name: str, value: float) -> Dict[str, Any]:
        """Interprets a single lab value against reference thresholds."""
        norm_key = test_name.lower().strip().replace(" ", "_").replace("-", "_")
        ref = cls.REFERENCE_RANGES.get(norm_key)

        if not ref:
            # Check partial match
            for k, v in cls.REFERENCE_RANGES.items():
                if k in norm_key or norm_key in k:
                    ref = v
                    break

        if not ref:
            return {
                "test_name": test_name,
                "value": value,
                "status": "UNKNOWN_TEST",
                "message": f"Reference range for '{test_name}' is not pre-configured."
            }

        status = "NORMAL"
        is_critical = False
        message = f"Within normal physiological range ({ref['min']} - {ref['max']} {ref['unit']})."

        if value < ref["min"]:
            status = "LOW"
            if value <= ref.get("critical_low", ref["min"]):
                status = "CRITICAL_LOW"
                is_critical = True
                message = f"CRITICAL LOW! Value {value} {ref['unit']} is below critical threshold {ref['critical_low']}."
            else:
                message = f"Below standard range ({ref['min']} - {ref['max']} {ref['unit']})."
        elif value > ref["max"]:
            status = "HIGH"
            if value >= ref.get("critical_high", ref["max"]):
                status = "CRITICAL_HIGH"
                is_critical = True
                message = f"CRITICAL HIGH! Value {value} {ref['unit']} is above critical threshold {ref['critical_high']}."
            else:
                message = f"Above standard range ({ref['min']} - {ref['max']} {ref['unit']})."

        return {
            "test_name": ref["name"],
            "value": value,
            "unit": ref["unit"],
            "reference_range": f"{ref['min']} - {ref['max']} {ref['unit']}",
            "status": status,
            "is_critical": is_critical,
            "interpretation": message
        }

    @classmethod
    def interpret_panel(cls, lab_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        """Interprets a full dictionary of lab tests."""
        results = []
        for test_name, val in lab_dict.items():
            results.append(cls.interpret(test_name, val))
        return results
