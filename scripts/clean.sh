#!/usr/bin/env bash

pkill -f -9 python
pkill -f -9 python3

# ls -l /dev/shm | grep nanovllm
rm -f /dev/shm/nanovllm*

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
  ipcrm -m $shmid
  echo "Deleted shared memory segment with ID: $shmid"
done

