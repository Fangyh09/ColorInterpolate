# ColorInterpolate
Interpolate color

## Usage
```python
w1 = np.random.rand(100, 100)
w2 = np.random.rand(100, 100)
w3 = np.random.rand(100, 100)

m1 = norm_color(w1, w2, w3)

plt.imsave("m1.png", m1)
```
