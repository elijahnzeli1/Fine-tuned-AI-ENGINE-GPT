import argparse
import tensorflow as tf
from models.fine_tuning.fine_tuning_pipeline import FineTuningPipeline
from utils.data_loader import load_json_data, save_json_data
from utils.model_utils import load_pretrained_model, compile_model

def main(args):
    # Load configuration
    config = load_json_data(args.config_file)

    # Load pretrained model
    base_model = load_pretrained_model(config['base_model'], trainable=False)

    # Create and compile the model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(config['num_classes'], activation='softmax')
    ])

    compile_model(
        model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Initialize fine-tuning pipeline
    pipeline = FineTuningPipeline(config, model)

    # Run fine-tuning pipeline
    performance = pipeline.run_pipeline(args.data_file)

    # Save performance metrics
    save_json_data(performance, args.output_file)

    print(f"Training completed. Performance metrics saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune models")
    parser.add_argument("--config_file", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--data_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save performance metrics")
    args = parser.parse_args()

    main(args)