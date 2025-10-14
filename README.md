# STATS 604 Project 3, Fall 2025

## Project Structure:
The project is organized with the following structure:

- lab-notebook.md : a markdown file which documents how the herb experiment was conducted
- pre-analysis_plan.md : a markdown file which documents the pre-analysis plan
- data/
    - raw_data/ : directory for all raw data files
    - processed_data/ : directory for all cleaned data
- figures/ : directory for all plots and generated figures
- results/ : directory for all results - either intermediate or final
- src/ : directory containing all code files

Additionally, all code in the src/ directory should only require files from
data/ or results/ and any intermediate output should be saved in one of those
two locations. This will make it easier to specify a dependency structure
in the Makefile to more efficiently produce the final report.
