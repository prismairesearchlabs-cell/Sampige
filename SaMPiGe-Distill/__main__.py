#!/usr/bin/env python3
"""
Main entry point for SaMPiGe-Distill
Allows running the package as a module: python -m SaMPiGe-Distill
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main as train_main
from validate import main as validate_main
from infer import main as infer_main


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SaMPiGe-Distill: Multi-Task Knowledge Distillation for Object Detection'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default=None, help='Path to config file')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate the model')
    val_parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    val_parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Data split')
    val_parser.add_argument('--save_results', action='store_true', help='Save results')
    val_parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    val_parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True, help='Input image, directory, or video')
    infer_parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    infer_parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    infer_parser.add_argument('--save_predictions', action='store_true', help='Save predictions')
    infer_parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    infer_parser.add_argument('--video', action='store_true', help='Input is video')
    infer_parser.add_argument('--frame_skip', type=int, default=1, help='Frame skip for video')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate main function
    if args.command == 'train':
        # Set config if provided
        if args.config:
            os.environ['CONFIG_PATH'] = args.config
        train_main()
    
    elif args.command == 'validate':
        # Convert args to validate format
        import types
        validate_args = types.SimpleNamespace(
            model_path=args.model_path,
            split=args.split,
            save_results=args.save_results,
            output_dir=args.output_dir,
            compare=args.compare
        )
        validate_main(validate_args)
    
    elif args.command == 'infer':
        # Convert args to infer format
        import types
        infer_args = types.SimpleNamespace(
            model_path=args.model_path,
            input=args.input,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_predictions=args.save_predictions,
            confidence=args.confidence,
            video=args.video,
            frame_skip=args.frame_skip
        )
        infer_main(infer_args)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
