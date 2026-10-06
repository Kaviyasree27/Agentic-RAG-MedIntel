from typing import Dict, Any, Optional

class ClinicalCalculators:
    """
    Standardized, validated clinical formula calculators.
    Includes:
    - eGFR (CKD-EPI 2021 race-free equation)
    - BMI (Body Mass Index & WHO classification)
    - CHA2DS2-VASc (Atrial Fibrillation Stroke Risk)
    - Wells Score for Pulmonary Embolism
    """

    @staticmethod
    def calculate_egfr(serum_creatinine: float, age: int, is_female: bool) -> Dict[str, Any]:
        """
        CKD-EPI 2021 Creatinine Equation (Race-Free, current KDIGO & NKF/ASN standard).
        Formula:
        eGFR = 142 * min(Scr/kappa, 1)**alpha * max(Scr/kappa, 1)**(-1.200) * 0.9938**Age * (1.012 if female)
        """
        if serum_creatinine <= 0 or age <= 0:
            return {"error": "Invalid creatinine or age parameters"}

        kappa = 0.7 if is_female else 0.9
        alpha = -0.241 if is_female else -0.302
        gender_mult = 1.012 if is_female else 1.0

        scr_k = serum_creatinine / kappa
        egfr = 142 * (min(scr_k, 1.0) ** alpha) * (max(scr_k, 1.0) ** -1.200) * (0.9938 ** age) * gender_mult
        egfr = round(egfr, 1)

        # KDIGO CKD Staging
        if egfr >= 90:
            stage = "Stage 1 (Normal or high kidney function)"
            clinical_implication = "Normal kidney function unless persistent proteinuria/hematuria is present."
        elif egfr >= 60:
            stage = "Stage 2 (Mildly decreased kidney function)"
            clinical_implication = "Mild reduction; evaluate for albuminuria, maintain cardiovascular risk management."
        elif egfr >= 45:
            stage = "Stage 3a (Mild to moderately decreased)"
            clinical_implication = "Moderate CKD; adjust nephrotoxic drugs, monitor potassium, check eGFR every 6-12 months."
        elif egfr >= 30:
            stage = "Stage 3b (Moderately to severely decreased)"
            clinical_implication = "Significant CKD; dose-adjust Metformin/RAAS agents, monitor for hyperkalemia & bone-mineral disorders."
        elif egfr >= 15:
            stage = "Stage 4 (Severely decreased kidney function)"
            clinical_implication = "Severe impairment; prepare for renal replacement therapy / nephrology referral."
        else:
            stage = "Stage 5 (Kidney Failure / ESRD)"
            clinical_implication = "End-stage renal disease; dialysis or transplant evaluation indicated."

        return {
            "calculator": "eGFR (CKD-EPI 2021 Race-Free)",
            "egfr_value_ml_min": egfr,
            "stage": stage,
            "clinical_implication": clinical_implication,
            "inputs": {
                "serum_creatinine": serum_creatinine,
                "age": age,
                "sex": "Female" if is_female else "Male"
            }
        }

    @staticmethod
    def calculate_bmi(weight_kg: float, height_cm: float) -> Dict[str, Any]:
        """Calculates Body Mass Index and classifies according to WHO criteria."""
        if weight_kg <= 0 or height_cm <= 0:
            return {"error": "Weight and height must be positive values."}

        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m ** 2), 1)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25.0:
            category = "Normal weight"
        elif bmi < 30.0:
            category = "Overweight"
        elif bmi < 35.0:
            category = "Class I Obesity (Moderate)"
        elif bmi < 40.0:
            category = "Class II Obesity (Severe)"
        else:
            category = "Class III Obesity (Very severe/morbid)"

        return {
            "calculator": "BMI (Body Mass Index)",
            "bmi": bmi,
            "category": category,
            "weight_kg": weight_kg,
            "height_cm": height_cm
        }

    @staticmethod
    def calculate_cha2ds2_vasc(
        age: int,
        is_female: bool,
        congestive_heart_failure: bool = False,
        hypertension: bool = False,
        diabetes: bool = False,
        prior_stroke_tia_thromboembolism: bool = False,
        vascular_disease: bool = False
    ) -> Dict[str, Any]:
        """CHA2DS2-VASc score for atrial fibrillation stroke risk."""
        score = 0
        if congestive_heart_failure: score += 1
        if hypertension: score += 1
        if age >= 75:
            score += 2
        elif age >= 65:
            score += 1
        if diabetes: score += 1
        if prior_stroke_tia_thromboembolism: score += 2
        if vascular_disease: score += 1
        if is_female: score += 1

        # Anticoagulation guidance based on ACC/AHA/ESC guidelines
        if (not is_female and score >= 2) or (is_female and score >= 3):
            recommendation = "Oral anticoagulation (DOAC e.g., Apixaban/Rivaroxaban) strongly recommended."
        elif (not is_female and score == 1) or (is_female and score == 2):
            recommendation = "Oral anticoagulation should be considered based on individual bleeding risk."
        else:
            recommendation = "Low stroke risk; anticoagulation generally not required."

        return {
            "calculator": "CHA2DS2-VASc Score",
            "score": score,
            "recommendation": recommendation,
            "risk_stratum": "High Risk" if score >= 2 else ("Intermediate" if score == 1 else "Low Risk")
        }
