import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.models.utils.eval_model_utils import aggregate_and_plot_metrics

if __name__ == "__main__":
    save_dir = sys.argv[1]

    print(f"Save dir: {save_dir}")
    
    try:
        average_metrics = aggregate_and_plot_metrics(save_dir=save_dir, postfix="_average")
        print(f"Averaged metrics and plots saved to: {save_dir}")
        print(f"Average metrics:\n{average_metrics}")
    except Exception as e:
        print(f"Error during evaluation {str(e)}")
        raise