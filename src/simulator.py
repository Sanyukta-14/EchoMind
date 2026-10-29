
import numpy as np
import pandas as pd


VARIABLES = [
    "temperature",
    "water_level",
    "plant_health",
    "food_supply",
    "animal_population",
]

# Animal population is the outcome, not a candidate fault cause.
FAULT_CAUSES = [
    variable for variable in VARIABLES
    if variable != "animal_population"
]

DEFAULT_FAULT_VALUES = {
    "temperature": 38.0,
    "water_level": 20.0,
    "plant_health": 20.0,
    "food_supply": 15.0,
}

VARIABLE_RANGES = {
    "temperature": (15.0, 40.0),
    "water_level": (0.0, 100.0),
    "plant_health": (0.0, 100.0),
    "food_supply": (0.0, 100.0),
    "animal_population": (0.0, 100.0),
}


def simulate_world(
    steps=200,
    fault_step=100,
    seed=42,
    intervention=None,
    fault_cause="water_level",
    fault_value=None,
):
    """
    Simulate an ecosystem with configurable faults and interventions.

    Temperature uses mean-reverting dynamics during normal operation.
    A temperature fault persists from fault_step onward.

    This is a synthetic model. Results do not establish real-world
    causation.
    """

    # --------------------------------------------------
    # 1. Validate simulation settings
    # --------------------------------------------------
    if (
        not isinstance(steps, (int, np.integer))
        or isinstance(steps, (bool, np.bool_))
        or steps < 2
    ):
        raise ValueError("steps must be an integer of at least 2.")

    if (
        not isinstance(fault_step, (int, np.integer))
        or isinstance(fault_step, (bool, np.bool_))
    ):
        raise ValueError("fault_step must be an integer.")

    steps = int(steps)
    fault_step = int(fault_step)

    if not 0 <= fault_step < steps:
        raise ValueError(
            "fault_step must be between 0 and steps - 1."
        )

    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
    ):
        raise ValueError("seed must be an integer.")

    seed = int(seed)

    # --------------------------------------------------
    # 2. Validate fault configuration
    # --------------------------------------------------
    if fault_cause not in FAULT_CAUSES:
        raise ValueError(
            f"Unsupported fault_cause: {fault_cause!r}. "
            f"Choose one of {FAULT_CAUSES}."
        )

    if fault_value is None:
        fault_value = DEFAULT_FAULT_VALUES[fault_cause]

    try:
        fault_value = float(fault_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fault_value must be numeric.") from exc

    if not np.isfinite(fault_value):
        raise ValueError("fault_value must be finite.")

    minimum, maximum = VARIABLE_RANGES[fault_cause]

    if not minimum <= fault_value <= maximum:
        raise ValueError(
            f"fault_value for {fault_cause!r} must be "
            f"between {minimum} and {maximum}."
        )

    # --------------------------------------------------
    # 3. Validate intervention
    # --------------------------------------------------
    if intervention is not None:
        if not isinstance(intervention, dict):
            raise TypeError(
                "intervention must be a dictionary or None."
            )

        intervention_variable = intervention.get("variable")

        if intervention_variable not in VARIABLES:
            raise ValueError(
                "Unknown intervention variable: "
                f"{intervention_variable!r}."
            )

        if "value" not in intervention:
            raise ValueError(
                "intervention must contain a 'value'."
            )

        try:
            intervention_value = float(intervention["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The intervention value must be numeric."
            ) from exc

        if not np.isfinite(intervention_value):
            raise ValueError(
                "The intervention value must be finite."
            )

        low, high = VARIABLE_RANGES[intervention_variable]

        if not low <= intervention_value <= high:
            raise ValueError(
                f"Intervention value for {intervention_variable!r} "
                f"must be between {low} and {high}."
            )

        # Copy the validated intervention without mutating the caller's dict.
        intervention = {
            "variable": intervention_variable,
            "value": intervention_value,
        }

    # --------------------------------------------------
    # 4. Initialize ecosystem
    # --------------------------------------------------
    rng = np.random.default_rng(seed)

    temperature = 25.0
    water_level = 80.0
    plant_health = 80.0
    food_supply = 70.0
    animal_population = 50.0

    records = []

    # --------------------------------------------------
    # 5. Run simulation
    # --------------------------------------------------
    for step in range(steps):

        # Mean-reverting temperature dynamics.
        # Normal temperature fluctuates around 25 rather than
        # wandering randomly toward its upper limit.
        temperature += (
            0.08 * (25.0 - temperature)
            + rng.normal(0, 0.6)
        )

        temperature = float(
            np.clip(temperature, 15.0, 40.0)
        )

        # A temperature fault persists from fault_step onward.
        if (
            fault_cause == "temperature"
            and step >= fault_step
        ):
            temperature = fault_value

        # Explicit intervention takes precedence.
        if (
            intervention is not None
            and step >= fault_step
            and intervention["variable"] == "temperature"
        ):
            temperature = intervention["value"]

        # --------------------------------------------------
        # Update water level
        # --------------------------------------------------
        # Preserve original ordering: inject water fault before
        # the normal water-level update.
        if (
            step == fault_step
            and fault_cause == "water_level"
        ):
            water_level = fault_value

        water_level += (
            0.8
            - max(0.0, temperature - 25.0) * 0.15
        )

        water_level = float(
            np.clip(water_level, 0.0, 100.0)
        )

        if (
            intervention is not None
            and step >= fault_step
            and intervention["variable"] == "water_level"
        ):
            water_level = float(
                np.clip(intervention["value"], 0.0, 100.0)
            )

        # --------------------------------------------------
        # Update plant health
        # --------------------------------------------------
        plant_health += (
            0.12 * water_level
            - 0.06 * temperature
            - 5.0
            + rng.normal(0, 1)
        )

        plant_health = float(
            np.clip(plant_health, 0.0, 100.0)
        )

        if (
            step == fault_step
            and fault_cause == "plant_health"
        ):
            plant_health = fault_value

        if (
            intervention is not None
            and step >= fault_step
            and intervention["variable"] == "plant_health"
        ):
            plant_health = float(
                np.clip(intervention["value"], 0.0, 100.0)
            )

        # --------------------------------------------------
        # Update food supply
        # --------------------------------------------------
        food_supply += (
            0.15 * plant_health
            - 8.0
            + rng.normal(0, 1)
        )

        food_supply = float(
            np.clip(food_supply, 0.0, 100.0)
        )

        if (
            step == fault_step
            and fault_cause == "food_supply"
        ):
            food_supply = fault_value

        if (
            intervention is not None
            and step >= fault_step
            and intervention["variable"] == "food_supply"
        ):
            food_supply = float(
                np.clip(intervention["value"], 0.0, 100.0)
            )

        # --------------------------------------------------
        # Update animal population
        # --------------------------------------------------
        # Animal population is the outcome variable.
        animal_population += (
            0.08 * food_supply
            - 5.0
            + rng.normal(0, 0.5)
        )

        animal_population = float(
            np.clip(animal_population, 0.0, 100.0)
        )

        if (
            intervention is not None
            and step >= fault_step
            and intervention["variable"] == "animal_population"
        ):
            animal_population = float(
                np.clip(intervention["value"], 0.0, 100.0)
            )

        # --------------------------------------------------
        # Record simulation state and fault metadata
        # --------------------------------------------------
        records.append({
            "step": step,
            "temperature": temperature,
            "water_level": water_level,
            "plant_health": plant_health,
            "food_supply": food_supply,
            "animal_population": animal_population,
            "fault_step": fault_step,
            "fault_cause": fault_cause,
            "fault_value": fault_value,
        })

    return pd.DataFrame(records)
