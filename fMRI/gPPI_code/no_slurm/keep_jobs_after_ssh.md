# Keep no-Slurm jobs running after you close SSH

`02_make_timecourse_files.sh`, `04_run_PPI_1st_level.sh`, `07_run_PPI_2nd_level_Replay.sh`, and `10_run_PPI_3rd_level.sh` run **in the foreground** of the shell that launched them (`xargs -P`, not Slurm). If that shell is your SSH session, closing CMD (or dropping Wi‑Fi) sends SIGHUP and **kills the jobs**.

Use **tmux on the HPC** (Wish), not on Windows. CMD is only the SSH client. Once the script is running *inside* tmux, you can close CMD; the jobs stay on the HPC.

## 1. Connect as usual

In CMD:

```bat
ssh hectoryam@wish
```

(Use whatever host/user you already use.)

## 2. Start a tmux session (on Wish)

```bash
which tmux || echo "tmux missing — try screen instead"
tmux new -s gppi04
```

Your prompt looks almost the same, but you are now inside a session named `gppi04`. If `tmux` is missing, use screen or `nohup` below.

Pick a session name per pipeline step if you run more than one (`gppi02`, `gppi04`, …).

## 3. In that tmux window, load the environment and start the jobs

Do this **inside tmux**. A new session does not keep modules/conda from an old SSH window unless you started tmux *after* loading them.

```bash
cd ~/cogitate-msp2/fMRI/gPPI_code/no_slurm/AT-seen-go

module load fsl/6.0.7.23
# conda activate cogitate_fmri   # if this step needs it
export BIDS_ROOT=/wishData/cogitate-data/fMRI/bids
export CODE_PATH="$HOME/cogitate-msp2/fMRI"
# export GPPI_NJOBS=8

bash 04_run_PPI_1st_level.sh
```

Leave it running. You should see `>> running N local job(s) with GPPI_NJOBS=...`.

## 4. Leave safely

**Option A — detach, then close CMD**

1. Press `Ctrl+b`, release both keys, then press `d`.
2. You should see `[detached (from session gppi04)]` and be back in a normal SSH shell.
3. You can `exit` SSH and close CMD.

**Option B — just close CMD** (or lose Wi‑Fi)

That only kills the SSH client. The tmux **server** on Wish keeps the session and the FEAT jobs. You do not have to detach first.

Do **not** run `04_run_PPI_1st_level.sh` (or 02 / 07 / 10) in the plain SSH shell before `tmux new`. Closing CMD will kill those jobs.

## 5. Reconnect later

In a new CMD window:

```bat
ssh hectoryam@wish
```

Then:

```bash
tmux ls
tmux attach -t gppi04
```

You should see the same script output, still scrolling if jobs are running, or the finished summary if they completed.

Detach again with `Ctrl+b` then `d` whenever you want to leave.

## Useful tmux commands (after `Ctrl+b`)

| Keys | What it does |
| --- | --- |
| `d` | Detach (jobs keep running) |
| `c` | New window in the same session |
| `n` / `p` | Next / previous window |
| `[` | Scrollback (arrows / PageUp; `q` to quit copy mode) |

```bash
tmux ls                          # list sessions
tmux attach -t gppi04            # reattach
tmux kill-session -t gppi04      # only after jobs are done
```

If attach says `session not found`, the session ended (script finished and you exited the shell inside tmux, or the machine rebooted). `tmux ls` will be empty.

## If tmux is not installed

### screen

```bash
screen -S gppi04
# load modules, run 04 ...
# detach: Ctrl+a, then d
```

Reconnect:

```bash
screen -ls
screen -r gppi04
```

### nohup (no live view)

```bash
cd ~/cogitate-msp2/fMRI/gPPI_code/no_slurm/AT-seen-go
nohup bash 04_run_PPI_1st_level.sh > ~/gppi04_nohup.log 2>&1 &
echo $!
disown
```

Close CMD. Later:

```bash
tail -f ~/gppi04_nohup.log
ps -u "$USER" | grep -E 'feat|xargs'
```

Prefer **tmux** so you can reattach and watch FEAT the same way you do in an interactive SSH session.
