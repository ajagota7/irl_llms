# Asymmetric Margin Loss Options for IRL

This document explains the different margin loss options available for Inverse Reinforcement Learning (IRL) training.

## Available Loss Functions

### 1. Symmetric Margin Loss (`max_margin`)
The original symmetric margin loss that enforces a fixed margin between detoxified and original rewards.

```yaml
training:
  irl_method: "max_margin"
  margin: 0.1  # Fixed margin value
```

**Loss Function:**
```
L = max(0, margin - (R(detoxified) - R(original)))
```

### 2. Asymmetric Margin Loss (`asymmetric_margin`)
Penalizes violations more heavily when the reward model assigns higher rewards to toxic outputs than non-toxic ones, reflecting the safety-critical nature of alignment tasks.

```yaml
training:
  irl_method: "asymmetric_margin"
  positive_penalty: 1.0  # penalty when detoxified > original (good case)
  negative_penalty: 2.0  # penalty when original > detoxified (bad case)
```

**Loss Function:**
```
L = -positive_penalty * (R(detoxified) - R(original)) if R(detoxified) > R(original)
L = -negative_penalty * (R(detoxified) - R(original)) if R(original) > R(detoxified)
```

This implements the asymmetric loss described in the paper:
```
L(x) = -x if x > 0
       -2x if x < 0
where x = R(detoxified) - R(original)
```

### 3. Confidence-Based Margin Loss (`confidence_margin`)
Uses a dynamic margin that increases with the confidence of the prediction, encouraging more confident separations.

```yaml
training:
  irl_method: "confidence_margin"
  base_margin: 0.1        # base margin value
  confidence_factor: 0.5  # factor to scale the dynamic margin
```

**Loss Function:**
```
dynamic_margin = base_margin + confidence_factor * |R(detoxified) - R(original)|
L = max(0, dynamic_margin - (R(detoxified) - R(original)))
```

### 4. Maximum Entropy IRL (`max_entropy`)
The original maximum entropy IRL loss function.

```yaml
training:
  irl_method: "max_entropy"
  temperature: 0.1  # temperature parameter for softmax
```

## Configuration Examples

### Example 1: Asymmetric Margin (Recommended for Safety)
```yaml
training:
  irl_method: "asymmetric_margin"
  positive_penalty: 1.0  # Standard penalty for good case
  negative_penalty: 2.0  # Double penalty for bad case (safety-critical)
```

### Example 2: Confidence-Based Margin
```yaml
training:
  irl_method: "confidence_margin"
  base_margin: 0.1
  confidence_factor: 0.5
```

### Example 3: Standard Symmetric Margin
```yaml
training:
  irl_method: "max_margin"
  margin: 0.1
```

## When to Use Each Loss Function

- **Symmetric Margin**: Good baseline, treats both directions equally
- **Asymmetric Margin**: Recommended for safety-critical applications where false positives (calling toxic content safe) are more dangerous than false negatives
- **Confidence Margin**: Useful when you want to encourage more confident predictions and better separation
- **Max Entropy**: Good for probabilistic interpretations of the reward function

## Safety Considerations

The asymmetric margin loss is particularly useful for safety-critical applications because:
1. It penalizes false positives (calling toxic content safe) more heavily
2. It reflects the asymmetric nature of safety - it's worse to miss toxic content than to be overly cautious
3. It aligns with the principle that safety violations should have higher penalties

## Usage

To use these loss functions, simply set the `irl_method` in your configuration file and provide the appropriate parameters. The training script will automatically use the correct loss function based on your configuration. 