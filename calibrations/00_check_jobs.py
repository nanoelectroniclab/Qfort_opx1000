"""
00_check_jobs.py
----------------
Check the status of all jobs currently or previously queued on the OPX1000.
Useful for diagnosing stuck jobs before running a new calibration.


"""
# %%
import sys
from datetime import timezone, timedelta
sys.path.insert(0, "..")

TWN = timezone(timedelta(hours=8))

from quam_config.my_quam import Quam

machine = Quam.load()
qmm = machine.connect()

jobs = qmm.get_jobs()
jobs = jobs[-5:]  # Show only the most recent 5 jobs

if not jobs:
    print("No jobs found.")
else:
    print(f"Found {len(jobs)} job(s) (showing last 5):\n")
    print(f"{'#':<4} {'Status':<12} {'Sim':<5} {'Started (UTC+8)':<22} {'Duration':<12} {'ID'}")
    print("-" * 95)
    for i, job in enumerate(jobs):
        m = job.metadata
        started  = m.started_at.astimezone(TWN).strftime("%Y-%m-%d %H:%M:%S") if m.started_at else "N/A"
        sim      = "Yes" if job.is_simulation else "No"
        if m.started_at and m.last_status_updated_at:
            delta = m.last_status_updated_at - m.started_at
            secs  = int(delta.total_seconds())
            duration = f"{secs//60}m {secs%60}s" if secs >= 60 else f"{secs}s"
        else:
            duration = "N/A"
        print(f"{i+1:<4} {str(job.status):<12} {sim:<5} {started:<22} {duration:<12} {job.id}")

    print()
    running = [j for j in jobs if str(j.status) in ("Running", "Pending", "Processing")]
    if running:
        print(f"!!! {len(running)} job(s) still active !!!")
        answer = input("Close all active program? (Y/n): ").strip().lower()
        if answer in ("y", "yes", ""):
            qmm.close_all_qms()
            print("✅ All program closed.")
        else:
            print("Skipped. Run 00_close_other_qms.py manually to clear them.")
    else:
        print("✅ No active jobs.")

# %%
