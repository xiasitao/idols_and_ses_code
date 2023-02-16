Codebase for the Bachelor's thesis "Idols and Socioeconomic Status" (Maximilian Schattauer, February 2023)

# Python Setup
- During development, the code was run with python 3.8.
- Dependencies are saved in the `poetry.lock` file to be used with poetry. My setup was inspired by: https://duncanleung.com/set-up-python-pyenv-virtualenv-poetry/

# Folders
- `data_exploration`: iPython scripts for data exploration
- `evaluation`: Standalone libraries and scripts for data inference and producing LaTeX-compatible plots and tables
- `tasks`: Data preparation tasks (to be used with pytask)
- `test`: Tests for the data preparation tasks