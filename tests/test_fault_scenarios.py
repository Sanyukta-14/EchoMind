from src.simulator import simulate_world


def main():
    causes = [
        "temperature",
        "water_level",
        "plant_health",
        "food_supply",
    ]

    for cause in causes:
        df = simulate_world(
            steps=200,
            fault_step=100,
            seed=42,
            fault_cause=cause,
        )

        print(f"\nFault cause: {cause}")
        print(f"Recorded cause: {df.loc[100, 'fault_cause']}")
        print(f"Recorded fault value: {df.loc[100, 'fault_value']}")
        print(f"Value at fault step: {df.loc[100, cause]}")


if __name__ == "__main__":
    main()