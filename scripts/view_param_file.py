# A quick script to print the contents (turned on experiments) of an Excel parameter file to the terminal
# Run with the following command:
# python -m scripts.view_param_file -p param_file.xlsx


from pprint import pprint
import pandas as pd

from utility import parameter_setup


def main(params_filename):
    assert params_filename.endswith('.xlsx')

    df = pd.read_excel(params_filename, engine='openpyxl')
    df = df[df['run'] == 1]

    print(f'\nNo of experiments: {len(df)}\n')

    # pprint(df)

    all_params = []
    for i in range(len(df)):
        params = df.iloc[i].to_dict()
        params = parameter_setup.parse_params(params)
        all_params.append(params)

    for i, param_set in enumerate(all_params):
        print(f'\n======================== Experiment {i+1} ({param_set["exp_id"]}) ========================\n')
        pprint(param_set)
        print(f'======================== End of Experiment {i+1} ========================\n')


if __name__ == '__main__':
    params_filename, other_args = parameter_setup.get_cmd_line_args()

    main(params_filename)
