#!/bin/bash
#this is a singularity problem only fixed recently
whoami=`whoami`
echo ${whoami}
echo Hello
mkdir -p /tmp/$(whoami)/singularity_cache
WHO=$(whoami)
SINGULARITY_CACHEDIR=/tmp/$(whoami)/singularity_cache
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
cd

command="$@"
echo "If you see the following error: \"container creation failed: mount /proc/self/fd/10->/var/singularity/mnt/session/rootfs error ...\" please just try again"
$sing run -B /home -B /eos -B /afs --bind /etc/krb5.conf:/etc/krb5.conf --bind /proc/fs/openafs/afs_ioctl:/proc/fs/openafs/afs_ioctl --bind /usr/vice/etc:/usr/vice/etc  /eos/home-j/jkiesele/singularity/geant4/geant4_latest.sif command="$@"

rm -rf /tmp/$(whoami)/singularity_cache

