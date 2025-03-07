import os
import shutil


def list_paths(
    directory: str,
    startswith: str = None,
    notstartswith: str = ".",
    sort: bool = True,
    full_path: bool = True,
) -> list[str]:
    filenames = sorted(os.listdir(directory))
    if startswith:
        filenames = [f for f in filenames if f.startswith(startswith)]
    if notstartswith:
        if type(notstartswith) is not str:
            notstartswith = [notstartswith]
        for string in notstartswith:
            filenames = [f for f in filenames if not f.startswith(string)]
    if not full_path:
        return filenames
    return [os.path.join(directory, f) for f in filenames]


def makedir_delete_if_exists(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
