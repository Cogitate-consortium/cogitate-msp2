import general_analysis
import specific_analysis
import quality_checks

"""
This simple module can run all the existing behavioral analyses:
- Quality checks (qc)
- General analysis (general)
- Specific analysis (specific)
Choose which of them you want to run (could be one, could be all). For general and specific, use the filter_general_qc
and filter_specific_qc parameters to choose if you want to analyze only subjects who pass the quality checks (True), 
or you want to analyze all subjects (False). Note that if you put "True" in any of the filters you don't have to 
mark the quality checks (qc) separatly.

@author: RonyHirsch
"""


def run_analyses(qc, general, filter_general_qc, specific, filter_specific_qc, data_path, save_path):
    if qc:
        quality_checks.check_data(data_path=data_path, plot_prescreen=True, save=True, save_path=save_path)
    if general:
        general_analysis.analyze(check=filter_general_qc, data_path=data_path, plot=True, save=True,
                                 save_path=save_path)
    if specific:
        specific_analysis.analyze(check=filter_specific_qc, data_path=data_path, plot=True, save=True,
                                  save_path=save_path)


if __name__ == "__main__":
    run_analyses(qc=False, general=True, filter_general_qc=True, specific=True, filter_specific_qc=True,
                 data_path=r"PATH_TO_FOLDER_WHICH_CONTAINS_ALL_SUBJECTS_BEH_DATA",
                 save_path=r"PATH_TO_WHERE_YOU_WANT_RESULTS_TO_BE_SAVED")
