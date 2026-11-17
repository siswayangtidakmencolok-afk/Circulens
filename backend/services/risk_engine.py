def calculate_contextual_risk(
    distribution,
    total,
    temperature=None,
    humidity="",
    harvest_age="",
    storage_location="",
):
    """
    Calculate contextual batch risk using:
    - visual quality distribution
    - storage temperature
    - humidity
    - harvest age
    - storage location

    This is a rule-based MVP risk engine.
    """

    if total <= 0:
        return {
            "risk_level": "unknown",
            "risk_score": 0,
            "risk_factors": [],
        }

    damaged_ratio = distribution.get("damaged", 0) / total
    dried_ratio = distribution.get("dried", 0) / total
    old_ratio = distribution.get("old", 0) / total

    deterioration_ratio = (
        damaged_ratio
        + dried_ratio
        + old_ratio
    )

    risk_score = 0
    risk_factors = []

    # ==================================================
    # 1. VISUAL QUALITY
    # ==================================================

    visual_score = deterioration_ratio * 70
    risk_score += visual_score

    if damaged_ratio > 0:
        risk_factors.append(
            f"{round(damaged_ratio * 100)}% of samples show visible damage."
        )

    if dried_ratio > 0:
        risk_factors.append(
            f"{round(dried_ratio * 100)}% of samples show signs of drying."
        )

    if old_ratio > 0:
        risk_factors.append(
            f"{round(old_ratio * 100)}% of samples are classified as old."
        )

    # ==================================================
    # 2. TEMPERATURE
    # ==================================================

    if temperature is not None:

        if temperature >= 30:
            risk_score += 20

            risk_factors.append(
                "High storage temperature may accelerate quality deterioration."
            )

        elif temperature >= 27:
            risk_score += 10

            risk_factors.append(
                "Storage temperature is relatively high and may increase deterioration risk."
            )

        elif temperature <= 10:
            risk_score += 3

            risk_factors.append(
                "Low storage temperature detected."
            )

    # ==================================================
    # 3. HUMIDITY
    # ==================================================

    humidity_normalized = str(
        humidity
    ).lower().strip()

    if humidity_normalized == "high":
        risk_score += 10

        risk_factors.append(
            "High humidity may increase the risk of quality deterioration."
        )

    elif humidity_normalized == "low":
        risk_score += 5

        risk_factors.append(
            "Low humidity may contribute to moisture loss and drying."
        )

    # ==================================================
    # 4. HARVEST AGE
    # ==================================================

    harvest_normalized = str(
        harvest_age
    ).lower().strip()

    if harvest_normalized in ["5-7", "7+"]:
        risk_score += 10

        risk_factors.append(
            "The batch has been stored for several days since harvest."
        )

    elif harvest_normalized == "3-4":
        risk_score += 5

        risk_factors.append(
            "The batch has been stored for multiple days since harvest."
        )

    # ==================================================
    # 5. STORAGE LOCATION
    # ==================================================

    storage_normalized = str(
        storage_location
    ).lower().strip()

    if storage_normalized in [
        "open",
        "transport",
        "market",
    ]:
        risk_score += 5

        risk_factors.append(
            "The current storage environment may provide less controlled conditions."
        )

    # ==================================================
    # FINAL SCORE
    # ==================================================

    risk_score = min(
        round(risk_score),
        100,
    )

    # ==================================================
    # RISK LEVEL
    # ==================================================

    if risk_score >= 60:
        risk_level = "high"

    elif risk_score >= 30:
        risk_level = "moderate"

    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
    }


def calculate_quality_score(
    distribution,
    total,
):
    """
    Calculate visual quality score.

    Unripe produce is treated as a maturity state,
    not direct quality deterioration.
    """

    if total <= 0:
        return 0

    deterioration_weights = {
        "damaged": 1.0,
        "dried": 0.65,
        "old": 0.45,
    }

    weighted_deterioration = sum(
        distribution.get(quality, 0) * weight
        for quality, weight in deterioration_weights.items()
    )

    deterioration_ratio = (
        weighted_deterioration / total
    )

    score = 100 - (
        deterioration_ratio * 100
    )

    return max(
        0,
        min(
            100,
            round(score),
        ),
    )


def calculate_at_risk_quantity(
    quantity,
    distribution,
    total,
    risk_level,
):
    """
    Estimate the quantity potentially at risk.

    The visual deterioration ratio is used
    as the baseline and contextual risk
    slightly adjusts the estimate.
    """

    if total <= 0 or quantity <= 0:
        return {
            "min": 0,
            "max": 0,
        }

    deterioration_count = (
        distribution.get("damaged", 0)
        + distribution.get("dried", 0)
        + distribution.get("old", 0)
    )

    deterioration_ratio = (
        deterioration_count / total
    )

    baseline = quantity * deterioration_ratio

    if risk_level == "high":
        adjustment = 1.10

    elif risk_level == "moderate":
        adjustment = 1.05

    else:
        adjustment = 1.00

    estimated = baseline * adjustment

    estimated = min(
        estimated,
        quantity,
    )

    return {
        "min": round(
            estimated * 0.80,
            2,
        ),
        "max": round(
            estimated,
            2,
        ),
    }


def generate_decision_recommendations(
    distribution,
    total,
    risk_level,
    temperature=None,
    humidity="",
    harvest_age="",
    storage_location="",
):
    """
    Generate actionable batch-level recommendations.

    This is a rule-based decision engine for the CircuLens MVP.

    The engine considers:
    - detected quality conditions
    - contextual risk level
    - temperature
    - humidity
    - harvest age
    - storage location

    Returns prioritized actions with:
    - priority
    - action
    - reason
    - pathway
    """

    if total <= 0:
        return []

    recommendations = []

    damaged_count = distribution.get("damaged", 0)
    dried_count = distribution.get("dried", 0)
    old_count = distribution.get("old", 0)
    ripe_count = distribution.get("ripe", 0)
    unripe_count = distribution.get("unripe", 0)

    damaged_ratio = damaged_count / total
    dried_ratio = dried_count / total
    old_ratio = old_count / total
    ripe_ratio = ripe_count / total
    unripe_ratio = unripe_count / total

    humidity_normalized = str(
        humidity
    ).lower().strip()

    harvest_normalized = str(
        harvest_age
    ).lower().strip()

    storage_normalized = str(
        storage_location
    ).lower().strip()

    # ==================================================
    # 1. DAMAGED PRODUCE
    # ==================================================

    if damaged_count > 0:

        if damaged_ratio >= 0.30:
            priority = 1
            action = "Separate and process damaged produce"
            reason = (
                f"{round(damaged_ratio * 100)}% of samples show visible damage. "
                "Separating them can reduce the chance of deterioration spreading "
                "to the rest of the batch."
            )

        else:
            priority = 2
            action = "Separate damaged produce"
            reason = (
                f"{round(damaged_ratio * 100)}% of samples show visible damage. "
                "These samples should be separated from the main batch."
            )

        recommendations.append({
            "priority": priority,
            "action": action,
            "reason": reason,
            "pathway": "Process",
        })

    # ==================================================
    # 2. HIGH RISK
    # ==================================================

    if risk_level == "high":

        recommendations.append({
            "priority": 1,
            "action": "Prioritize this batch for immediate handling",
            "reason": (
                "The contextual risk level is high. "
                "Further delay may increase potential quality loss."
            ),
            "pathway": "Process",
        })

    # ==================================================
    # 3. DRIED / OLD
    # ==================================================

    if dried_count > 0 or old_count > 0:

        aging_count = dried_count + old_count
        aging_ratio = aging_count / total

        if aging_ratio >= 0.30:
            priority = 1 if risk_level in ["moderate", "high"] else 2
        else:
            priority = 3

        recommendations.append({
            "priority": priority,
            "action": "Prioritize aging produce for processing or preservation",
            "reason": (
                f"{round(aging_ratio * 100)}% of samples are classified as "
                "dried or old and may have reduced suitability for fresh sale."
            ),
            "pathway": "Preserve",
        })

    # ==================================================
    # 4. UNRIPE PRODUCE
    # ==================================================

    if unripe_count > 0:

        if unripe_ratio >= 0.50:
            priority = 2
        else:
            priority = 3

        recommendations.append({
            "priority": priority,
            "action": "Hold suitable unripe produce for maturity progression",
            "reason": (
                f"{round(unripe_ratio * 100)}% of samples are classified as "
                "unripe. Suitable produce may require additional time before "
                "fresh distribution."
            ),
            "pathway": "Mature",
        })

    # ==================================================
    # 5. RIPE PRODUCE
    # ==================================================

    if ripe_count > 0:

        if ripe_ratio >= 0.50:
            priority = 1
        else:
            priority = 3

        recommendations.append({
            "priority": priority,
            "action": "Prioritize suitable ripe produce for fresh distribution",
            "reason": (
                f"{round(ripe_ratio * 100)}% of samples are classified as ripe "
                "and may be suitable for fresh distribution."
            ),
            "pathway": "Sell Fresh",
        })

    # ==================================================
    # 6. HIGH HUMIDITY
    # ==================================================

    if humidity_normalized == "high":

        recommendations.append({
            "priority": 2 if risk_level == "high" else 3,
            "action": "Improve humidity control",
            "reason": (
                "High humidity may increase deterioration risk. "
                "Improve ventilation or move the batch to a more suitable "
                "storage environment when possible."
            ),
            "pathway": "Storage",
        })

    # ==================================================
    # 7. HIGH TEMPERATURE
    # ==================================================

    if temperature is not None and temperature >= 27:

        recommendations.append({
            "priority": 2 if temperature >= 30 else 3,
            "action": "Move the batch to a cooler storage environment",
            "reason": (
                f"Current storage temperature is {temperature}°C. "
                "Higher temperatures may accelerate quality deterioration."
            ),
            "pathway": "Storage",
        })

    # ==================================================
    # 8. HARVEST AGE
    # ==================================================

    if harvest_normalized in ["5-7", "7+"]:

        recommendations.append({
            "priority": 2,
            "action": "Prioritize older batches before newer batches",
            "reason": (
                "This batch has been stored for several days since harvest "
                "and should receive higher handling priority."
            ),
            "pathway": "Prioritize",
        })

    # ==================================================
    # 9. STORAGE ENVIRONMENT
    # ==================================================

    if storage_normalized in [
        "open",
        "transport",
        "market",
    ]:

        recommendations.append({
            "priority": 3,
            "action": "Move the batch to a more controlled storage environment",
            "reason": (
                "The current storage environment may expose the produce "
                "to less controlled conditions."
            ),
            "pathway": "Storage",
        })

    # ==================================================
    # 10. SORT / MIXED BATCH
    # ==================================================

    quality_categories_present = sum([
        damaged_count > 0,
        dried_count > 0,
        old_count > 0,
        ripe_count > 0,
        unripe_count > 0,
    ])

    if quality_categories_present >= 2:

        recommendations.append({
            "priority": 2,
            "action": "Sort the batch by quality condition",
            "reason": (
                "Multiple quality conditions were detected. "
                "Sorting allows each group to receive a more appropriate "
                "handling pathway."
            ),
            "pathway": "Sort",
        })

    # ==================================================
    # REMOVE DUPLICATE ACTIONS
    # ==================================================

    unique_recommendations = []

    seen_actions = set()

    for recommendation in recommendations:

        action = recommendation["action"]

        if action in seen_actions:
            continue

        seen_actions.add(action)

        unique_recommendations.append(
            recommendation
        )

    # ==================================================
    # SORT BY PRIORITY
    # ==================================================

    unique_recommendations.sort(
        key=lambda item: item["priority"]
    )

    # ==================================================
    # RETURN TOP ACTIONS
    # ==================================================

    return unique_recommendations[:6]