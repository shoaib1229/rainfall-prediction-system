"""
Rainfall Prediction CLI & Inference Interface.
Allows users to input weather measurements interactively or via CLI flags,
and outputs rainfall predictions and probabilities using the serialized model.
"""
import sys
import argparse
from src.predict import load_model, predict_single, EXPECTED_FEATURES

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rainfall Prediction Inference Tool — Predict rain occurrence from weather measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--humidity", type=float, default=None, help="Relative humidity (percentage, e.g. 85.0)")
    parser.add_argument("--cloud", type=float, default=None, help="Cloud cover (percentage, e.g. 75.0)")
    parser.add_argument("--sunshine", type=float, default=None, help="Sunshine duration (hours, e.g. 2.5)")
    parser.add_argument("--wind-speed", type=float, default=None, help="Wind speed (km/h, e.g. 18.5)")
    parser.add_argument("--dew-point", type=float, default=None, help="Dew point temperature (°C, e.g. 21.0)")
    parser.add_argument("--pressure", type=float, default=None, help="Atmospheric pressure (hPa, e.g. 1011.2)")
    parser.add_argument("--wind-direction", type=float, default=None, help="Wind direction (degrees, e.g. 80.0)")
    parser.add_argument("--model-path", type=str, default="models/rainfall_prediction_model.pkl",
                        help="Path to serialized model bundle")
    return parser.parse_args()

def prompt_float(prompt_text, default_val=None):
    """
    Prompts user for numeric input with validation and optional default.
    """
    while True:
        try:
            val_str = input(prompt_text).strip()
            if not val_str and default_val is not None:
                return default_val
            if not val_str:
                print("  [Warning] Value left blank. Will use model training median.")
                return None
            return float(val_str)
        except ValueError:
            print("  [Error] Invalid input. Please enter a valid numeric value.")

def run_interactive(bundle):
    print("\n" + "=" * 55)
    print("      RAINFALL PREDICTION — INTERACTIVE MODE")
    print("=" * 55)
    print("Enter the weather measurements for today:")
    print("(Press Enter without value to use median imputation)\n")
    
    input_data = {}
    input_data['humidity'] = prompt_float("  Humidity (%): ")
    input_data['cloud'] = prompt_float("  Cloud cover (%): ")
    input_data['sunshine'] = prompt_float("  Sunshine (hours): ")
    input_data['windspeed'] = prompt_float("  Wind speed (km/h): ")
    input_data['dewpoint'] = prompt_float("  Dew point (°C): ")
    input_data['pressure'] = prompt_float("  Atmospheric pressure (hPa): ")
    input_data['winddirection'] = prompt_float("  Wind direction (degrees): ")
    
    result = predict_single(bundle, input_data)
    print_prediction_result(result)

def print_prediction_result(result):
    print("\n" + "-" * 55)
    print("                PREDICTION RESULT")
    print("-" * 55)
    print(f"  Rain Prediction : {result['rain_label']}")
    print(f"  Rain Probability: {result['probability_percent']}")
    print(f"  Model Engine    : {result['model_name']}")
    print("-" * 55)

def main():
    args = parse_arguments()
    
    # Load serialized model bundle
    try:
        bundle = load_model(args.model_path)
    except Exception as e:
        print(f"[Error] Failed to load model bundle from '{args.model_path}': {e}", file=sys.stderr)
        sys.exit(1)
        
    # Check if CLI feature arguments were provided
    cli_args_provided = any([
        args.humidity is not None,
        args.cloud is not None,
        args.sunshine is not None,
        args.wind_speed is not None,
        args.dew_point is not None,
        args.pressure is not None,
        args.wind_direction is not None
    ])
    
    if cli_args_provided:
        input_data = {
            'humidity': args.humidity,
            'cloud': args.cloud,
            'sunshine': args.sunshine,
            'windspeed': args.wind_speed,
            'dewpoint': args.dew_point,
            'pressure': args.pressure,
            'winddirection': args.wind_direction
        }
        try:
            result = predict_single(bundle, input_data)
            print_prediction_result(result)
        except Exception as e:
            print(f"[Error] Inference failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # No CLI arguments -> run interactive mode
        run_interactive(bundle)

if __name__ == "__main__":
    main()
