"""Watch the training process tree and failures in Ray background threads."""
import json
import re
import time

import psutil

FATAL = re.compile(
    r'RuntimeError:.*(?:Connection closed by peer|Connection reset by peer)'
    r'|[1-9][0-9]* Workers .*killed due to memory pressure'
    r'|ray\.exceptions\.(?:OutOfMemoryError|ActorDiedError|RayActorError)')


def stop_tree(processes):
    # psutil Process retains creation time and protects against reused PIDs.
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=15)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def supervise(child, logfile, root):
    parent = psutil.Process(child.pid)
    known = {parent}
    pressure_polls = 0
    failure = None
    with logfile.open(
            errors='replace') as log, (root /
                                       'memory.jsonl').open('a') as memory:
        while child.poll() is None:
            try:
                known.update(parent.children(recursive=True))
            except psutil.NoSuchProcess:
                break
            vm = psutil.virtual_memory()
            workers = []
            for process in known:
                try:
                    name = process.name()
                    if 'EnvGroup' in name or 'FluxEnvWorker' in name:
                        workers.append({
                            'pid': process.pid,
                            'rss_bytes': process.memory_info().rss
                        })
                except psutil.NoSuchProcess:
                    pass
            memory.write(
                json.dumps({
                    'time': time.time(),
                    'host_used_percent': vm.percent,
                    'host_available_bytes': vm.available,
                    'env_workers': workers
                }) + '\n')
            memory.flush()
            match = FATAL.search(log.read())
            pressure_polls = pressure_polls + 1 if vm.percent >= 85 else 0
            if match:
                failure = match.group(0)
            elif pressure_polls >= 3:
                failure = ('Host RAM usage >=85% for three consecutive '
                           '15-second samples')
            if failure:
                (root / 'failure.json').write_text(
                    json.dumps({
                        'time': time.time(),
                        'reason': failure
                    },
                               indent=2) + '\n')
                stop_tree(list(known))
                break
            time.sleep(15)
    # Catch fatal output even if the driver exited between polling samples.
    with logfile.open(errors='replace') as log:
        match = FATAL.search(log.read())
        if match and failure is None:
            failure = match.group(0)
    code = child.wait()
    stop_tree(list(known))
    return code, failure
