import argparse

def convert_args_to_dict(args: argparse.Namespace) -> dict:
    '''
    From an argparse Namespace, create a dictionnary with only inputed CLI
    arguments. Allow to use with default values in functions.
    '''
    args_dict = {
        argument[0]: argument[1]
        for argument
        in args._get_kwargs()
        if argument[1] is not None}

    return args_dict
