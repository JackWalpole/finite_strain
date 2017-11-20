# finite_strain
## Jack Walpole

Routines to enable the calculation of the finite strain ellipsoid from velocity gradient tensors.

Example code:

```python
>>> import finite_strain as fs
>>> a = fs.Path('files/_L_3555_25_25')
>>> f = a.accumulate_strain()
>>> f.plot()
```
