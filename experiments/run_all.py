"""Run the whole empirical pipeline in dependency order, a few jobs at a time.

A step is skipped when its marker file results/pipeline/<name>.done exists, so
the script can be restarted after an interruption.  Logs go to results/<log>.
usage: python experiments/run_all.py [--jobs 3]
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MARK = os.path.join(ROOT, "results", "pipeline")
PY = sys.executable


def steps():
    s = []

    def add(name, cmd, deps=(), log=None, threads=1):
        s.append(dict(name=name, cmd=cmd, deps=list(deps), log=log or f"{name}.log", threads=threads))

    add("build_panel", "experiments/build_panel.py", threads=2)
    add("build_global", "experiments/build_global.py", threads=2)
    for w in (7, 14, 28, 56):
        add(f"us_w{w}", f"experiments/backtest.py --window {w}", ["build_panel"], f"run_window{w}.log")
        add(f"us_w{w}_var", f"experiments/backtest.py --window {w} --variants", ["build_panel"],
            f"run_window{w}variants.log")
    for t in ("0.5", "0.8", "0.9", "0.95"):
        add(f"us_tau{t}", f"experiments/backtest.py --window 28 --tau {t}", ["build_panel"], f"run_window28tau{t}.log")
        add(f"us_tau{t}_var", f"experiments/backtest.py --window 28 --tau {t} --variants", ["build_panel"],
            f"run_window28tau{t}variants.log")
    add("chronos", "experiments/chronos_candidate.py", ["build_global"], threads=3)
    add("weather_gfs", "experiments/weather_candidate.py --source gfs --threads 2", ["build_global"],
        "weather_gfs.log", threads=2)
    add("weather_exact", "experiments/weather_candidate.py --noise 0 --threads 2", ["build_global"],
        "weather_noise0.log", threads=2)
    add("weather_plus", "experiments/weather_plus_candidate.py --threads 2", ["build_global"],
        "weather_plus.log", threads=2)
    g = ["chronos"]
    add("gl_none", "experiments/backtest_global.py --window 28", g, "global_w28.log")
    add("gl_gfs", "experiments/backtest_global.py --window 28 --weather gfs", g + ["weather_gfs"],
        "global_w28_weather-gfs.log")
    add("gl_exact", "experiments/backtest_global.py --window 28 --weather exact", g + ["weather_exact"],
        "global_w28_weather-exact.log")
    plus = g + ["weather_gfs", "weather_plus"]
    add("gl_plus", "experiments/backtest_global.py --window 28 --weather plus", plus, "global_w28_weather-plus.log")
    add("gl_plus_w14", "experiments/backtest_global.py --window 14 --weather plus", plus + ["gl_plus"],
        "global_w14_weather-plus.log")
    add("gl_plus_tau0.9", "experiments/backtest_global.py --window 28 --tau 0.9 --weather plus", plus + ["gl_plus"],
        "global_w28_tau0.9_weather-plus.log")
    add("gl_plus_tau0.99", "experiments/backtest_global.py --window 28 --tau 0.99 --weather plus",
        plus + ["gl_plus"], "global_w28_tau0.99_weather-plus.log")
    add("ng_plus", "experiments/neural_gate.py --window 28 --weather plus", ["gl_plus"],
        "neuralgate_w28_weather-plus.log")
    add("ng_gfs", "experiments/neural_gate.py --window 28 --weather gfs", ["gl_gfs"], "neuralgate_w28_weather-gfs.log")
    add("ng_none", "experiments/neural_gate.py --window 28", ["gl_none"], "neuralgate_w28.log")
    add("storage", "experiments/storage_siting.py", ["gl_plus"], "storage_weather-plus.log")
    add("simulation", "experiments/simulation.py", [], "simulation.log")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(MARK, exist_ok=True)
    todo = steps()
    done = {x["name"] for x in todo if os.path.exists(os.path.join(MARK, x["name"] + ".done"))}
    failed, running = set(), {}
    while True:
        for name, (p, st) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[name]
            if rc == 0:
                open(os.path.join(MARK, name + ".done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
                done.add(name)
                print(time.strftime("%H:%M:%S"), "done", name, flush=True)
            else:
                failed.add(name)
                print(time.strftime("%H:%M:%S"), "FAILED", name, "exit", rc, flush=True)
        ready = [x for x in todo if x["name"] not in done | failed | set(running)
                 and all(d in done for d in x["deps"])]
        blocked = [x for x in todo if x["name"] not in done | failed | set(running)
                   and any(d in failed for d in x["deps"])]
        for x in blocked:
            failed.add(x["name"])
            print(time.strftime("%H:%M:%S"), "SKIPPED (dependency failed)", x["name"], flush=True)
        used = sum(st["threads"] for _, st in running.values())
        for x in ready:
            if len(running) >= args.jobs or used + x["threads"] > args.jobs + 1:
                break
            env = dict(os.environ, OMP_NUM_THREADS=str(x["threads"]), THREADS=str(x["threads"]),
                       TAG="global_w28_taucalibrated_weather-plus")
            log = open(os.path.join(ROOT, "results", x["log"]), "w")
            p = subprocess.Popen([PY, "-u"] + x["cmd"].split(), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[x["name"]] = (p, x)
            used += x["threads"]
            print(time.strftime("%H:%M:%S"), "start", x["name"], flush=True)
        if not running and not ready:
            break
        time.sleep(10)
    print("finished; done:", len(done), "failed:", sorted(failed), flush=True)


if __name__ == "__main__":
    main()
