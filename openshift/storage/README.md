# Storage

Persistent storage provisioning for the OpenShift ML cluster.

## Tier2 PVC (one-time setup)

Each user needs their own CephFS PVC for shared, persistent storage across
jobs. This is where model caches, datasets, and benchmark results live.

Replace `<YOUR_NAME>` in `tier2.yml` and apply once:

```bash
# Edit the PVC name
sed -i 's/<YOUR_NAME>/jdoe/' tier2.yml

# Create it
oc apply -f tier2.yml
```

The PVC is `ReadWriteMany`, so it can be mounted by multiple pods
simultaneously. All benchmark Jobs and utility pods mount it at `/tier2`.

### What lives on tier2

| Path | Purpose |
|------|---------|
| `/tier2/hf-hub` | HuggingFace model cache (`HF_HOME`) |
| `/tier2/benchmark_results/` | Timestamped benchmark output directories |
| `/tier2/data/` | Shared datasets |
| `/tier2/uv-cache/` | uv package cache |

## Storage tiers

| Tier | Type | Speed | Persistence | Typical use |
|------|------|-------|-------------|-------------|
| tier0 | Node-local hostPath (`/var/mnt/scratch/scratch`) | Fastest | Survives pod restarts, shared across pods on the same node | Scratch space |
| tier1 | Ephemeral LVM NVMe (`lvms-h100-tier1-storage`) | Fast | Pod-lifetime only, deleted when pod terminates | Model copies for fast loading |
| tier2 | CephFS (`ocs-storagecluster-cephfs-tier2`) | Network | Persistent across pods and nodes | Model cache, results, datasets |
