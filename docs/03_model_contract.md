# Model Contract

The preferred V1 production contract is a hint U-Net:

```json
{
  "model_type": "hint_unet",
  "input_size": 768,
  "input_channels": 5,
  "output_channels": 3,
  "input_range": "0_1",
  "output_range": "0_1",
  "expects_lineart_inverted": false
}
```

Input channels:

- `0`: grayscale clean line art.
- `1`: hint red.
- `2`: hint green.
- `3`: hint blue.
- `4`: hint mask.

Output:

- RGB color in `0..1`, shape `[1, 3, H, W]`.

