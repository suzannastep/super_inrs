import subprocess
import time
import sys

# Capture all command-line arguments except the script name
args = sys.argv[1:]

def submit_job(file,params):
    res = subprocess.run(['sbatch','/home/sueparkinson/deeprelu/super_inrs/phantom_exp/srun.sh','python',file] + params, capture_output=True, text=True)
    job_id = None
    if res.stdout:
        # Typical sbatch output: "Submitted batch job 123456"
        job_id = int(res.stdout.strip().split()[-1])
    return job_id

def wait_for_job(job_id,freq=1):
    while True:
        res = subprocess.run(['squeue', '-j', str(job_id)], capture_output=True, text=True)
        # If job is not found in squeue, it's done (finished/failed/cancelled)
        if str(job_id) not in res.stdout:
            break
        time.sleep(freq*60)  # Poll every freq minutes

if __name__ == "__main__":
    
    file = "/home/sueparkinson/deeprelu/super_inrs/phantom_exp/train.py"
    params = sys.argv[1:]

    print("parameters to pass onto train.py:",params)

    job_id = submit_job(file,params)
    print("Started",job_id)
    wait_for_job(job_id)
    print(f"Job {job_id} finished.")