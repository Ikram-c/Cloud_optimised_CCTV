# Maintenance breakdown - cctv_zarr on a GCE trial instance

What runs itself, what needs a human, and on what cadence. Times
assume the stock `config.gce.yaml` deployment (loopback panel,
SSH-tunnel access, mock archive, daily prune timer).

## Automatic - no action required

| Mechanism | What it does |
|---|---|
| `cctv-zarr.service` | Panel auto-starts on boot, restarts on failure, capped at 700 MB RAM |
| `cctv-zarr-prune.timer` | Daily retention run: expires chunks/stores/exports/cache, journals every deletion |
| Upload cap (`ui.max_upload_mb`) | Bounds each incoming video, browser- and server-side |
| Source quarantine | Ingested originals move to `_ingested/`, expired by the sweep |
| Upload/deletion journals | Resumable transfers; provable erasure |
| Swap file | Absorbs memory spikes on the 1 GB instance |

## Weekly (~2 minutes, in the SSH window)

```bash
systemctl status cctv-zarr --no-pager     # panel healthy?
systemctl list-timers | grep prune        # prune ran and is scheduled?
df -h /                                   # disk trend
tail -5 /opt/cctv_zarr/data/stores/deletion_log.jsonl
tail -5 /opt/cctv_zarr/data/stores/access_log.jsonl
```

Healthy looks like: service `active (running)`, the timer showing a
recent `PASSED` and a future `NEXT`, disk well under 50%, deletion
entries appearing on schedule. Review the access log for anything
you do not recognise - it is the Art. 5(2) accountability trail.

## Monthly (~15 minutes)

```bash
sudo apt-get update && sudo apt-get upgrade -y   # OS security patches
sudo reboot                                       # picks up kernel updates
```

After the reboot confirm `systemctl status cctv-zarr` again (it is
enabled, so it should return on its own). Rotate the panel access
code if one is set (`security.auth_token` in
`/opt/cctv_zarr/config.yaml`, then `sudo systemctl restart
cctv-zarr`). Skim `deletion_log.jsonl` against your retention
policy - this is the evidence a regulator asks for. Check Billing
-> Budgets in the Console for surprises (a correctly configured
deployment on the free tier should trend at ~$0 after trial
credit).

Enable unattended security updates once and the apt step mostly
disappears:

```bash
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## Quarterly / as needed

- **Software updates**: upload a new `cctv_zarr.zip`, unzip over
  the old tree, re-run `deploy/gce/setup_gce.sh` (it rebuilds the
  venv and re-copies units), `sudo systemctl restart cctv-zarr`.
  Your `config.yaml` is not overwritten (`cp -n`).
- **Governance review**: does the `governance:` block still name
  the right controller, purpose, and retention policy?
- **Key rotation**: if `encrypt_archive` is on, generate a new
  Fernet key, re-upload stores, retire the old key.
- **Restore drill**: run one `cctv-zarr query --movement-only`
  against the archive to prove fetch still works end to end.

## Trial-account lifecycle

The trial gives $300 of credit for 90 days (verify current terms
at signup). When it ends, the VM is **stopped, not deleted** - the
disk survives for a grace period. To keep running past the trial,
activate full billing; an e2-micro with a 30 GB standard disk in a
free-tier US region then continues at $0/month under the Always
Free tier (egress beyond the free allowance bills normally - the
SSH-tunnel workflow keeps egress tiny). Set a budget alert
(Billing -> Budgets & alerts, e.g. $5) on day one so nothing
surprises you.

## Failure playbook

| Symptom | Fix |
|---|---|
| Panel not answering | `sudo systemctl restart cctv-zarr`; then `journalctl -u cctv-zarr -n 50` for the cause |
| Disk filling | `sudo systemctl start cctv-zarr-prune.service` now; check `_ingested/` and `stores/exports/`; shorten the retention hours in config |
| Ingest stuck/slow | Expected on shared vCPU for big files - watch the panel's progress; check `journalctl -u cctv-zarr -f` |
| VM unreachable | Console -> VM instances -> Reset; inspect the serial console log |
| Wrong footage ingested | `cctv-zarr erase <store> --start ... --end ... --archive --cache` (journalled) |
| Suspected exposure | Rotate `auth_token`, review `access_log.jsonl`, remember Arts. 33-34 notification duties are the operator's |
| Lost access code | Edit `security.auth_token` in config.yaml over SSH, restart the service |

## What this deployment never does by itself

Ingest without being asked, expose a public port, upload real
footage to a real bucket (mock archive default), or delete
anything without a journal entry. Anything outside that envelope
is a change a human made - which is exactly what the logs are for.
