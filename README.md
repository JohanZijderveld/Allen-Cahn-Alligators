# Allen-Cahn Equation

Code for the MATH2731 Allen-Cahn group project.

To draw plots include the following lines at the start
```python
import allen_cahn as ac
x, U, E, saved = ac.solve("semi", h=0.005, tau=0.01)
```