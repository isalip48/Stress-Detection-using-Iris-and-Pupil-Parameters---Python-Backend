"""
Custom Layers for Dual-Stream Age-Aware Model
Extracted from: Dual_Stream_Age_Aware_Training.ipynb
Author: AI Assistant
Date: 2025-11-11

Contains custom Keras layers used in the stress detection model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class WeightedFeatureFusion(layers.Layer):
    """
    Learnable weighted fusion of two feature streams with dynamic per-sample alpha.
    
    Architecture:
        - Gating network predicts fusion weight (alpha) per sample
        - Alpha ∈ [0, 1] via sigmoid activation
        - Fusion: fused = alpha * iris_features + (1 - alpha) * pupil_age_features
    
    Interpretation:
        α → 1.0: Iris stream (tension rings) is more important for this image
        α → 0.0: Pupil+age stream (dilation) is more important for this image
        α ≈ 0.5: Both streams equally important for this image
    
    From main notebook Section 8.3
    """
    
    def __init__(self, name='weighted_fusion', **kwargs):
        super(WeightedFeatureFusion, self).__init__(name=name, **kwargs)
        
        # Gating network: learns to predict alpha per sample
        self.gating_network = tf.keras.Sequential([
            layers.Dense(8, activation='relu', name=f'{name}_gate_dense_1'),
            layers.Dense(1, activation='sigmoid', name=f'{name}_gate_output_alpha')
        ], name=f'{name}_gating_network')
        
        # Store the last computed alpha for monitoring
        self.last_alpha = None
    
    def call(self, inputs):
        """
        Args:
            inputs: [pupil_age_features, iris_features]
                pupil_age_features: (batch, feature_dim) - Pupil+Age context
                iris_features: (batch, feature_dim) - Iris patterns
        
        Returns:
            fused_features: (batch, feature_dim) - Weighted combination
        """
        pupil_age_features, iris_features = inputs
        
        # 1. Concatenate both feature streams for gating network
        concatenated_features = layers.concatenate([iris_features, pupil_age_features])
        
        # 2. Predict dynamic alpha for each sample (shape: batch_size, 1)
        # KEY: Alpha is computed PER SAMPLE!
        alpha = self.gating_network(concatenated_features)
        
        # Store alpha for monitoring purposes
        self.last_alpha = alpha
        
        # 3. Apply dynamic, per-sample weighted fusion
        fused_features = (alpha * iris_features) + ((1.0 - alpha) * pupil_age_features)
        
        return fused_features
    
    def get_config(self):
        config = super(WeightedFeatureFusion, self).get_config()
        return config


class EdgeAttentionModule(layers.Layer):
    """
    Novel Component #1: Edge Attention Module
    **EXACT MATCH to training notebook Section 5.1!**
    
    Purpose:
        Learn to focus on edge features (tension ring boundaries, pupil edges).
        Uses spatial attention to weight important edge regions.
    
    Output: 16 channels (expanded from 1 edge channel)
    """
    
    def __init__(self, name='edge_attention', **kwargs):
        super(EdgeAttentionModule, self).__init__(name=name, **kwargs)
        
        # Attention network
        self.attention_conv = layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding='same',
            activation='sigmoid',
            name=f'{name}_attention_conv'
        )
        
        # Feature processing (outputs 16 channels)
        self.feature_conv = layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{name}_feature_conv'
        )
    
    def call(self, inputs):
        """
        Args:
            inputs: [edge_channel, rgb_channels]
                edge_channel: (batch, H, W, 1) - Edge features
                rgb_channels: (batch, H, W, 3) - RGB features
        
        Returns:
            attended_features: (batch, H, W, 16)
        """
        edge_channel, rgb_channels = inputs
        
        # Concatenate for attention
        concat = layers.Concatenate(axis=-1)([edge_channel, rgb_channels])
        
        # Compute attention weights
        attention_weights = self.attention_conv(concat)
        
        # Apply attention to RGB
        attended_rgb = layers.Multiply()([rgb_channels, attention_weights])
        
        # Combine edge with attended RGB
        combined = layers.Concatenate(axis=-1)([edge_channel, attended_rgb])
        
        # Process attended features (outputs 16 channels)
        attended_features = self.feature_conv(combined)
        
        return attended_features
    
    def get_config(self):
        config = super(EdgeAttentionModule, self).get_config()
        return config


class FeatureAttentionModule(layers.Layer):
    """
    Novel Component #2: Feature Attention Module
    **EXACT MATCH to training notebook Section 5.1!**
    
    Purpose:
        Learn to focus on texture features (BlackHat morphological features).
        Uses channel attention to weight important texture patterns.
    
    Output: 16 channels (expanded from 1 texture channel)
    """
    
    def __init__(self, name='feature_attention', **kwargs):
        super(FeatureAttentionModule, self).__init__(name=name, **kwargs)
        
        # Channel attention network
        self.global_pool = layers.GlobalAveragePooling2D(name=f'{name}_global_pool')
        
        self.fc1 = layers.Dense(
            units=8,
            activation='relu',
            name=f'{name}_fc1'
        )
        
        self.fc2 = layers.Dense(
            units=4,  # 4 channels: RGB(3) + Texture(1)
            activation='sigmoid',
            name=f'{name}_fc2'
        )
        
        # Feature processing (outputs 16 channels)
        self.feature_conv = layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{name}_feature_conv'
        )
    
    def call(self, inputs):
        """
        Args:
            inputs: [texture_channel, rgb_channels]
                texture_channel: (batch, H, W, 1) - Texture features
                rgb_channels: (batch, H, W, 3) - RGB features
        
        Returns:
            attended_features: (batch, H, W, 16)
        """
        texture_channel, rgb_channels = inputs
        
        # Concatenate
        concat = layers.Concatenate(axis=-1)([texture_channel, rgb_channels])
        
        # Channel attention
        pooled = self.global_pool(concat)
        channel_weights = self.fc1(pooled)
        channel_weights = self.fc2(channel_weights)
        channel_weights = layers.Reshape((1, 1, 4))(channel_weights)
        
        # Apply attention
        attended = layers.Multiply()([concat, channel_weights])
        
        # Process attended features (outputs 16 channels)
        attended_features = self.feature_conv(attended)
        
        return attended_features
    
    def get_config(self):
        config = super(FeatureAttentionModule, self).get_config()
        return config


# Custom objects dictionary for model loading
CUSTOM_OBJECTS = {
    'WeightedFeatureFusion': WeightedFeatureFusion,
    'EdgeAttentionModule': EdgeAttentionModule,
    'FeatureAttentionModule': FeatureAttentionModule
}


if __name__ == "__main__":
    print("🧪 Testing Custom Layers...")
    print("\n✅ Available layers:")
    print("  - WeightedFeatureFusion: Dynamic per-sample fusion")
    print("  - EdgeAttentionModule: Edge feature attention")
    print("  - FeatureAttentionModule: Texture feature attention")
    print("\nLayers are ready for use in model loading!")
