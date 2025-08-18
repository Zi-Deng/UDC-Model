#!/usr/bin/env python3
"""
Debug script for ConvNext model issues
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.convnext import ConvNextConfig, ConvNextForImageClassification
from model.ResNet import ResNetConfig, ResNetForImageClassification
from utils.loss_functions import LossFunctions

def test_model_creation(model_type="convnext"):
    """Test model creation and basic functionality"""
    print(f"\n=== Testing {model_type.upper()} Model ===")
    
    try:
        if model_type.lower() == "convnext":
            config = ConvNextConfig(num_labels=2)
            model = ConvNextForImageClassification(config)
        else:
            config = ResNetConfig(num_labels=2, depths=[3, 4, 6, 3])
            model = ResNetForImageClassification(config)
        
        print(f"✓ Created {model_type} model with {config.num_labels} labels")
        
        # Test forward pass
        batch_size = 2
        channels = 3
        height = 224
        width = 224
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        print(f"✓ Created dummy input: {dummy_input.shape}")
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output type: {type(outputs)}")
        print(f"✓ Has logits attribute: {hasattr(outputs, 'logits')}")
        
        if hasattr(outputs, 'logits'):
            print(f"✓ Logits shape: {outputs.logits.shape}")
            print(f"✓ Logits type: {type(outputs.logits)}")
            print(f"✓ Logits device: {outputs.logits.device}")
            
            # Test loss functions
            dummy_labels = torch.randint(0, 2, (batch_size,))
            print(f"✓ Created dummy labels: {dummy_labels}")
            
            loss_functions = LossFunctions()
            
            # Test different loss functions
            for loss_name in ["cross_entropy", "seesaw"]:
                try:
                    if loss_name == "cross_entropy":
                        loss = loss_functions.cross_entropy(outputs.logits, dummy_labels)
                    elif loss_name == "seesaw":
                        loss = loss_functions.seesaw_loss(outputs.logits, dummy_labels)
                    else:
                        loss_fn = loss_functions.loss_function(loss_name)
                        loss = loss_fn(outputs.logits, dummy_labels)
                    print(f"✓ {loss_name} loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"✗ {loss_name} loss failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("✗ No logits attribute in outputs")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test loading the fixed config"""
    print("\n=== Testing Config Loading ===")
    
    import json
    
    try:
        with open("config/convnext_fixed.json", "r") as f:
            config = json.load(f)
        
        print("✓ Config loaded successfully")
        print(f"✓ Model type: {config.get('model_type', 'NOT SPECIFIED!')}")
        print(f"✓ Num labels: {config.get('num_labels')}")
        print(f"✓ Loss function: {config.get('loss_function')}")
        
        cost_matrix = config.get('cost_matrix', [])
        if cost_matrix:
            rows = len(cost_matrix)
            cols = len(cost_matrix[0]) if cost_matrix else 0
            print(f"✓ Cost matrix size: {rows}x{cols}")
            
            if rows != config.get('num_labels') or cols != config.get('num_labels'):
                print(f"⚠️  Warning: Cost matrix ({rows}x{cols}) doesn't match num_labels ({config.get('num_labels')})")
            else:
                print("✓ Cost matrix dimensions match num_labels")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 ConvNext Debug Script")
    print("=" * 50)
    
    config_ok = test_config_loading()
    
    if config_ok:
        convnext_ok = test_model_creation("convnext")
        resnet_ok = test_model_creation("resnet")
        
        if convnext_ok and resnet_ok:
            print("\n🎉 All tests passed! ConvNext should work correctly.")
        else:
            print("\n❌ Some tests failed. Check the errors above.")
    else:
        print("\n❌ Config loading failed. Fix the config file first.")