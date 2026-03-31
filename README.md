# Speed Restriction Economic Impact Calculator (SREIC)

This repository contains the Python kinematic model used in the Speed Restriction Economic Impact Calculator (SREIC), a decision-support framework for rail asset management in Finland.

## Main file
- `model.py`: self-contained Python module for restriction-induced journey-time impact calculation

## Purpose
The model estimates additional running time caused by local speed restrictions using a three-phase structure:
- braking
- restricted-speed plateau
- re-acceleration

## Example
Run:

```bash
python model.py
