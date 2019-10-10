import subprocess
from datetime import datetime
import json
import code_util.constants as const

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def save_new_run_info():
    run_start_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    github_hash = get_git_revision_hash()
    benchmark_score = -1

    run_info = {
            'run_start_time':run_start_time,
            'description':'TODO',
            'github_hash': github_hash.decode("utf-8"),
            'benchmark_score':benchmark_score
        }

    with open(const.run_info_path + 'run_info.json', 'w') as json_file:
        json.dump(run_info, json_file)


