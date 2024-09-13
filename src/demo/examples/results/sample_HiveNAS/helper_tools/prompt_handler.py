import os
from ipywidgets import widgets
from IPython.display import display
from IPython.display import clear_output
"""User prompts-handler
"""

class PromptHandler:
    '''Wrapper for input prompt-handling methods
    '''

    @staticmethod
    def setup_ipy_widgets(on_yaml_export, on_yaml_init):
        ''' Sets up configuration loader/exporter IPython widgets \
        (Google Colab-exclusive).

        Args:
            on_yaml_export (func): action to execute upon \
            yaml export button click
            on_yaml_init (func): action to execute upon yaml \
            load button click
        '''

        path_textarea = widgets.Text(placeholder='/path/to/config.yaml',
                                     value='/content/config.yaml')
        export_btn = widgets.Button(description="Export *Form* Config")
        load_btn = widgets.Button(description="Load Config")
        output = widgets.Output()

        def on_export_click(b):
            path = path_textarea.value

            filename = os.path.basename(path)
            path = os.path.dirname(path)

            with output:
                clear_output(wait=True)
                if filename == '':
                    print('\nInvalid path / filename!\n\n')
                on_yaml_export(path, filename)

        def on_load_click(b):
            path = path_textarea.value

            with output:
                clear_output(wait=True)
                on_yaml_init(path)


        export_btn.on_click(on_export_click)
        load_btn.on_click(on_load_click)

        bot_box = widgets.HBox([load_btn, export_btn])
        cont_box = widgets.VBox([path_textarea, bot_box])

        display(cont_box, output)


    @staticmethod
    def prompt_yes_no(question, default='y'):
        '''Yes/no query; reverts to default value if no input is given

        Args:
            question (str): printed prompt question
            default (str, optional): user answer to revert to if no \
            response is given (empty input) ; defaults to "yes"

        Returns:
            bool: user response
        '''

        valid_res = {
            'yes': True,
            'y': True,
            'no': False,
            'n': False
        }

        choice = None

        while choice not in valid_res:
            choice = input(f'{question} (y/n): ').lower().replace(' ', '') or default

        return valid_res[choice]


