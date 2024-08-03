# postMerger
**postMerger** provides surrogate fits for binary black-hole remnants.

## Available models

- `3dq8_20M`: Ringdown amplitudes from a non-precessing, quasi-circular, binary black-hole merger. Calibrated at $20M$ after the peak of $|h_{22}|$.

## Usage

```python
import postMerger
```

### List of available fits

```python
postMerger.allowed_fits
>>> ['3dq8_20M']
```

### Load fit

```python
fitname = '3dq8_20M'
fit = = postMerger.load_fit(fitname)
```

## Evaluate QNMs

We provide a QNM evaluator. See the corresponding [example notebook](examples/qnm_Kerr.ipynb) for advanced usage.

```python
## mass and spin
mass = 67
spin = 0.67

## mode
mode = (2,2,0)

## evaluate in SI units
f, tau = postMerger.qnm_Kerr(mass,spin,mode,SI_units=True)

## results
print('frequency (Hz): ',f)
print('damping time (s): ',tau)
>>> frequency (Hz):  250.71404280475124
>>> damping time (s):  0.004032098030215414
```

